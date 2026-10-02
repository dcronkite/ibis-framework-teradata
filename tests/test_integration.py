"""Opt-in tests. All created tables are session-local and uniquely named."""

import datetime
import os
from collections import Counter
from decimal import Decimal

import ibis
import pyarrow as pa
import pytest

from ibis_framework_teradata import Backend

pytestmark = pytest.mark.integration


@pytest.fixture
def td():
    host = os.environ.get('IBIS_TERADATA_TEST_HOST')
    if not host:
        pytest.skip('Set IBIS_TERADATA_TEST_HOST to enable live Teradata tests')
    options = {'host': host, 'batch_size': 2}
    for key in ('user', 'password', 'database', 'logmech', 'dbs_port'):
        if value := os.environ.get(f'IBIS_TERADATA_TEST_{key.upper()}'):
            options[key] = value
    with Backend().connect(**options) as backend:
        yield backend


def test_local_remote_join_preserves_duplicates_and_null_semantics(td):
    source = pa.table({
        'id': [1, 2, None],
        'amount': pa.array(
            [Decimal('10.50'), Decimal('20.00'), Decimal('99.00')],
            type=pa.decimal128(18, 2),
        ),
    })
    name = ibis.util.gen_name('td_remote')
    remote = td.create_table(name, source, temp=True)
    local = ibis.memtable(pa.table(
        {'id': [1, 1, 3, None], 'label': ['first', 'second', 'missing', 'null']}
    ))
    joined = remote.join(local, 'id').select(remote.id, remote.amount, local.label)
    result = td.to_pyarrow(joined).to_pylist()
    expected = [
        {'id': 1, 'amount': Decimal('10.50'), 'label': 'first'},
        {'id': 1, 'amount': Decimal('10.50'), 'label': 'second'},
    ]
    assert Counter(tuple(row.items()) for row in result) == Counter(
        tuple(row.items()) for row in expected
    )
    assert Counter(
        tuple(row.items()) for row in td.to_pyarrow(joined).to_pylist()
    ) == Counter(tuple(row.items()) for row in expected)
    # commit must retain the session's lookup rows
    td.con.commit()
    assert td.to_pyarrow(joined).num_rows == 2
    td.cleanup_memtables()
    assert td.to_pyarrow(joined).num_rows == 2


def test_polars_memtable_roundtrip(td):
    pl = pytest.importorskip('polars')
    local = ibis.memtable(
        pl.DataFrame({
            'row_id': [0, 1, 2],
            'id': [1, 1, None],
            'text': ['é', '中文', None],
            'active': [True, False, None],
        })
    )
    assert td.to_polars(local.order_by('row_id')).to_dict(as_series=False) == {
        'row_id': [0, 1, 2],
        'id': [1, 1, None],
        'text': ['é', '中文', None],
        'active': [True, False, None],
    }


@pytest.mark.parametrize('output', ['pandas', 'arrow', 'polars'])
def test_varchar_roundtrip_preserves_whitespace_and_nbsp(td, output):
    values = ['  a   ', 'b\u00a0', '\u00a0c ', ' ', '', None]
    local = ibis.memtable(pa.table({'row_id': range(len(values)), 'text': values}))
    expr = local.order_by('row_id')
    if output == 'pandas':
        result = td.execute(expr)['text']
        assert result.isna().iloc[-1]
        actual = [*result.iloc[:-1].tolist(), None]
    elif output == 'arrow':
        actual = td.to_pyarrow(expr)['text'].to_pylist()
    else:
        pytest.importorskip('polars')
        actual = td.to_polars(expr)['text'].to_list()
    assert actual == values


def test_char_export_padding_and_explicit_sql_normalization(td):
    name = ibis.util.gen_name('td_char_padding')
    with td._safe_raw_sql(
            f'CREATE MULTISET VOLATILE TABLE "{name}" ('
            'fixed CHAR(8) CHARACTER SET UNICODE CASESPECIFIC, '
            'varying VARCHAR(8) CHARACTER SET UNICODE CASESPECIFIC) '
            'NO PRIMARY INDEX ON COMMIT PRESERVE ROWS'
    ):
        pass
    with td._safe_raw_sql(f'INSERT INTO "{name}" VALUES (\'a\', \'a\')'):
        pass
    query = f'SELECT fixed, varying FROM "{name}"'
    source = td.sql(query, schema={'fixed': 'string', 'varying': 'string'})
    row = td.to_pyarrow(source).to_pylist()[0]
    assert row['fixed'].startswith('a')
    # export width depends on the server/user settings
    assert len(row['fixed']) >= 8
    assert set(row['fixed'][1:]) <= {' '}
    assert row['varying'] == 'a'
    assert td.to_pyarrow(source.select(fixed=source.fixed.rstrip())).to_pylist() == [
        {'fixed': 'a'}
    ]
    normalized = td.sql(
        'SELECT TRIM(TRAILING \' \' FROM fixed) AS trimmed, '
        'CAST(fixed AS VARCHAR(8)) AS casted '
        f'FROM "{name}"',
        schema={'trimmed': 'string', 'casted': 'string'},
    )
    normalized_row = td.to_pyarrow(normalized).to_pylist()[0]
    assert normalized_row['trimmed'] == 'a'
    # cast avoids export padding
    assert normalized_row['casted'].rstrip(' ') == 'a'
    assert len(normalized_row['casted']) <= 8


def test_decimal_boolean_date_timestamp_and_binary_roundtrip(td):
    data = pa.table({
        'row_id': [0, 1],
        'amount': pa.array(
            [Decimal('999999999999.120000'), None], type=pa.decimal128(24, 6)
        ),
        'active': [True, None],
        'date': [datetime.date(2026, 1, 31), None],
        'timestamp': pa.array(
            [datetime.datetime(2026, 1, 31, 12, 1, 2, 123456), None],
            # noqa: DTZ001 -- tests timezone-naive storage
            type=pa.timestamp('us'),
        ),
        'bytes': [b'\x00\xff', None],
    })
    local = ibis.memtable(data)
    assert td.to_pyarrow(local.order_by('row_id')).to_pylist() == data.to_pylist()
    assert td.to_pyarrow(
        local.order_by('row_id').select(negated=~local.active)
    ).to_pylist() == [{'negated': False}, {'negated': None}]


def test_empty_memtable_and_typed_nulls(td):
    empty = ibis.memtable(pa.table({'id': pa.array([], type=pa.int64())}))
    result = td.to_pyarrow(empty)
    assert result.schema == empty.schema().to_pyarrow() and result.num_rows == 0
    nulls = ibis.memtable(pa.table({'id': pa.array([None, None], type=pa.int64())}))
    assert td.to_pyarrow(nulls).to_pylist() == [{'id': None}, {'id': None}]


def test_metadata_and_namespace(td):
    name = ibis.util.gen_name('td_metadata')
    table = td.create_table(
        name,
        schema={'tiny': 'int8', 'amount': 'decimal(18,2)', 'timestamp': 'timestamp(3)'},
        temp=True,
    )
    # explicit sql reflects native cursor metadata
    reflected = td.sql(f'SELECT * FROM "{name}"')
    assert reflected.schema() == table.schema()
    assert isinstance(td.current_database, str)
    assert isinstance(td.version, str)


def test_calendar_quarters_and_limits(td):
    local = ibis.memtable(
        pa.table({
            'id': [1, 2, 3],
            'date': [datetime.date(2026, 1, 31)] * 3,
        })
    )
    expr = (
        local.order_by(local.id)
        .limit(1, offset=1)
        .select('id', future=local.date + ibis.interval(quarters=1))
    )
    assert td.to_pyarrow(expr).to_pylist() == [
        {'id': 2, 'future': datetime.date(2026, 4, 30)}
    ]


@pytest.mark.parametrize('temporal_type', ['date', 'timestamp'])
def test_integer_literal_lookback_between_filter(td, temporal_type):
    base = (
        datetime.date(2026, 1, 15)
        if temporal_type == 'date'
        else datetime.datetime(2026, 1, 15, 12, 30)  # noqa: DTZ001 -- tests timezone-naive timestamps
    )
    offsets = [-7, 0, -8, 1, -5, None, -5, -7]
    source = pa.table(
        {
            'id': [0, 1, 2, 3, 4, 5, 6, 0],
            'dt': [
                base + datetime.timedelta(days=days) if days is not None else None
                for days in offsets
            ],
            'index_date': [base] * 6 + [None, base],
        }
    )
    name = ibis.util.gen_name('td_lookback')
    table = td.create_table(name, source, temp=True)
    lookback = ibis.literal(7).as_interval('D')
    expr = table.filter(
        table.dt.between(table.index_date - lookback, table.index_date)
    ).select('id')
    # between includes both bounds, excludes nulls, and retains duplicate rows
    assert Counter(td.to_pyarrow(expr)['id'].to_pylist()) == Counter([0, 0, 1, 4])


@pytest.mark.parametrize('unit', [
    'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us',
])
def test_integer_column_timestamp_interval_arithmetic(td, unit):
    base = datetime.datetime(2026, 1, 31, 12, 30)  # noqa: DTZ001 -- tests timezone-naive timestamps
    calendar_values = {
        'Y': (base.replace(year=2027), base.replace(year=2025)),
        'Q': (base.replace(month=4, day=30), base.replace(year=2025, month=10)),
        'M': (base.replace(month=2, day=28), base.replace(year=2025, month=12)),
    }
    durations = {
        'W': datetime.timedelta(weeks=1),
        'D': datetime.timedelta(days=1),
        'h': datetime.timedelta(hours=1),
        'm': datetime.timedelta(minutes=1),
        's': datetime.timedelta(seconds=1),
        'ms': datetime.timedelta(milliseconds=1),
        'us': datetime.timedelta(microseconds=1),
    }
    future, past = (
        calendar_values[unit]
        if unit in calendar_values
        else (base + durations[unit], base - durations[unit])
    )
    table = ibis.memtable(
        pa.table({'id': [0, 1, 2, 3], 'n': [1, -1, 0, None], 'ts': [base] * 4})
    )
    expr = table.order_by('id').select(result=table.ts + table.n.as_interval(unit))
    assert td.to_pyarrow(expr)['result'].to_pylist() == [future, past, base, None]


def test_reconnect_reuploads_the_same_expression(td):
    local = ibis.memtable(pa.table({'id': [1, 1, None]}))
    before = td.to_pyarrow(local).to_pylist()
    td.reconnect()
    assert Counter(row['id'] for row in td.to_pyarrow(local).to_pylist()) == Counter(
        row['id'] for row in before
    )
