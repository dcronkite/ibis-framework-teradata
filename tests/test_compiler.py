import sqlite3
from decimal import Decimal

import ibis
import ibis.common.exceptions as exc
import pytest
import sqlglot as sg

from conftest import memtable
from ibis_framework_teradata import Backend


@pytest.fixture
def compiler():
    return Backend()


@pytest.fixture
def table():
    return ibis.table({
        'id': 'int64',
        'amount': 'decimal(18,2)',
        'label': 'string',
        'flag': 'boolean',
        'ts': 'timestamp',
    },
        name='sales',
    )


def test_remote_memtable_join_compiles_without_connection(compiler, table):
    local = memtable(
        {'id': [1, 1, None], 'category': ['a', 'b', 'c']},
        schema={'id': 'int64', 'category': 'string'},
        name='lookup',
    )
    joined = table.left_join(local, 'id')
    expr = joined.group_by('category').aggregate(
        total=joined.amount.sum(), count=joined.count()
    )
    sql = compiler.compile(expr)
    assert 'LEFT OUTER JOIN "lookup"' in sql
    assert 'GROUP BY' in sql and 'SUM(' in sql
    assert 'VALUES' not in sql
    assert sg.parse_one(sql, read=compiler.dialect) is not None


def test_parameter_is_substituted_and_escaped(compiler, table):
    parameter = ibis.param('string')
    sql = compiler.compile(
        table.filter(table.label == parameter), params={parameter: 'O\'Reilly'}
    )
    assert 'O\'\'Reilly' in sql
    assert 'CASE WHEN' not in sql  # Keep ordinary filter predicates sargable.


@pytest.mark.parametrize('build', [
    lambda t: t.select(t.label.lower(), t.label.length()),
    lambda t: t.select(
        t.label.contains('x'), t.label.startswith('%'), t.label.endswith('_')
    ),
    lambda t: t.select(t.label.substr(1, 3), t.label.replace('x', 'y')),
    lambda t: t.select(t.amount.fill_null(Decimal('0.00')), t.id.cast('string')),
    lambda t: t.select(t.flag.ifelse('yes', 'no')),
    lambda t: t.select(ibis.cases((t.id > 1, 'a'), else_='b')),
    lambda t: t.select(t.ts.year(), t.ts.month(), t.ts.quarter()),
    lambda t: t.select((t.ts + ibis.interval(quarters=1)).name('next')),
    lambda t: t.group_by(t.label).aggregate(
        total=t.amount.sum(where=t.flag), any=t.flag.any(), all=t.flag.all()
    ),
    lambda t: t.mutate(row=ibis.row_number().over(ibis.window(order_by=t.id))),
    lambda t: t.order_by(t.id).limit(2, offset=1),
    lambda t: t.filter(t.id.isin([1, 2, None])),
    lambda t: t.filter(t.label.ilike('a%')),
])
def test_etl_expressions_compile(compiler, table, build):
    sql = compiler.compile(build(table))
    assert sg.parse_one(sql, read='teradata') is not None
    assert 'BOOL_AND' not in sql and 'BOOL_OR' not in sql


def test_quarter_uses_calendar_months(compiler, table):
    sql = compiler.compile(table.ts + ibis.interval(quarters=1))
    assert 'ADD_MONTHS' in sql and ', 3)' in sql and '90' not in sql


def test_standalone_quarter_interval_uses_three_months(compiler):
    sql = compiler.compile(ibis.interval(quarters=1))
    assert 'MONTH' in sql and '90' not in sql


@pytest.mark.parametrize('temporal_type', ['date', 'timestamp'])
@pytest.mark.parametrize('days_lookback', [0, 7, 365, -7])
def test_integer_literal_lookback_between_filter(compiler, temporal_type, days_lookback):
    table = ibis.table({'dt': temporal_type, 'index_date': temporal_type}, name='events')
    lookback = ibis.literal(days_lookback).as_interval('D')
    expr = table.filter(table.dt.between(table.index_date - lookback, table.index_date))
    sql = compiler.compile(expr)
    assert f'BETWEEN "t0"."index_date" - CAST({days_lookback} AS INTERVAL DAY(4)) ' 'AND "t0"."index_date"' in sql
    assert sg.parse_one(sql, read=compiler.dialect) is not None


def test_parameterized_lookback_filter(compiler):
    table = ibis.table({'dt': 'date', 'index_date': 'date'}, name='events')
    days = ibis.param('int64')
    lookback = days.as_interval('D')
    expr = table.filter(table.dt.between(table.index_date - lookback, table.index_date))
    assert 'CAST(14 AS INTERVAL DAY(4))' in compiler.compile(expr, params={days: 14})


@pytest.mark.parametrize(('unit', 'expected'), [
    ('Y', 'NUMTOYMINTERVAL((CAST("t0"."n" AS BIGINT)) * 12, \'MONTH\')'),
    ('Q', 'NUMTOYMINTERVAL((CAST("t0"."n" AS BIGINT)) * 3, \'MONTH\')'),
    ('M', 'NUMTOYMINTERVAL("t0"."n", \'MONTH\')'),
    ('W', 'CAST((CAST("t0"."n" AS BIGINT)) * 7 AS INTERVAL DAY(4))'),
    ('D', 'CAST("t0"."n" AS INTERVAL DAY(4))'),
    ('h', 'NUMTODSINTERVAL("t0"."n", \'HOUR\')'),
    ('m', 'NUMTODSINTERVAL("t0"."n", \'MINUTE\')'),
    ('s', 'NUMTODSINTERVAL("t0"."n", \'SECOND\')'),
    ('ms', 'NUMTODSINTERVAL((CAST("t0"."n" AS DECIMAL(20, 0))) * 0.001'),
    ('us', 'NUMTODSINTERVAL((CAST("t0"."n" AS DECIMAL(20, 0))) * 0.000001'),
])
def test_integer_column_interval_conversion(compiler, unit, expected):
    table = ibis.table({'n': 'int64'}, name='events')
    sql = compiler.compile(table.select(delta=table.n.as_interval(unit)))
    assert expected in sql
    assert sg.parse_one(sql, read=compiler.dialect) is not None


@pytest.mark.parametrize('unit', ['Y', 'Q', 'M'])
@pytest.mark.parametrize('subtract', [False, True])
def test_integer_calendar_arithmetic_uses_add_months(compiler, unit, subtract):
    table = ibis.table({'n': 'int64', 'dt': 'date'}, name='events')
    interval = table.n.as_interval(unit)
    expr = table.dt - interval if subtract else table.dt + interval
    sql = compiler.compile(expr)
    assert 'ADD_MONTHS(' in sql and 'AS INT)' in sql
    assert 'NUMTOYMINTERVAL' not in sql
    if unit in ('Y', 'Q'):
        factor = 12 if unit == 'Y' else 3
        assert f') * {factor}' in sql
    sign = '-' if subtract else ''
    assert f'ADD_MONTHS("t0"."dt", {sign}CAST(' in sql
    assert sg.parse_one(sql, read='teradata') is not None


def test_null_integer_interval_stays_null(compiler):
    interval = ibis.literal(None, type='int64').as_interval('D')
    sql = compiler.compile(interval)
    assert 'NULL' in sql and 'AS INTERVAL DAY(4)' in sql
    assert sg.parse_one(sql, read=compiler.dialect) is not None


@pytest.mark.parametrize('interval_type', [
    'INTERVAL DAY(4)',
    'INTERVAL DAY',
    'INTERVAL DAY(4) TO SECOND(6)',
    'INTERVAL YEAR(4) TO MONTH',
    'INTERVAL MONTH(4)',
    'INTERVAL HOUR(4) TO MINUTE',
    'INTERVAL MINUTE(4)',
    'INTERVAL SECOND(4, 6)',
])
def test_native_interval_type_precision_roundtrip(compiler, interval_type):
    sql = f'SELECT CAST(n AS {interval_type}) AS delta FROM events'
    parsed = sg.parse_one(sql, read=compiler.dialect)
    assert parsed.sql(dialect=compiler.dialect) == sql


@pytest.mark.parametrize('interval', [
    ibis.literal(1).as_interval('ns'), ibis.interval(nanoseconds=1)
])
def test_nanosecond_intervals_fail_with_a_precision_error(compiler, interval):
    with pytest.raises(exc.UnsupportedOperationError, match='microsecond'):
        compiler.compile(interval)


def test_integer_division_keeps_fractional_result(compiler):
    t = ibis.table({'a': 'int64', 'b': 'int64'}, name='data')
    with sqlite3.connect(':memory:') as con:
        con.execute('CREATE TABLE data (a INTEGER, b INTEGER)')
        con.executemany('INSERT INTO data VALUES (?,?)', [(1, 2), (-3, 2)])
        assert con.execute(
            compiler.compile(t.select(ratio=t.a / t.b, floor=t.a // t.b))
        ).fetchall() == [(0.5, 0), (-1.5, -2)]


def test_integer_aggregations_widen_inputs(compiler):
    t = ibis.table({'id': 'int32', 'flag': 'boolean'}, name='data')
    sql = compiler.compile(
        t.aggregate(total=t.id.sum(), average=t.id.mean(), count=t.count(), minimum=t.flag.min())
    )
    assert 'SUM(CAST("t0"."id" AS BIGINT))' in sql
    assert 'AVG(CAST("t0"."id" AS FLOAT))' in sql
    assert 'CAST(COUNT(*) AS BIGINT)' in sql
    assert 'BOOL_AND' not in sql


def test_string_concat_and_join_null_semantics(compiler):
    t = ibis.table({'a': 'string', 'b': 'string'}, name='data')
    expr = t.select(concat=t.a + t.b, joined=ibis.literal('/').join([t.a, t.b]))
    sql = compiler.compile(expr)
    assert 'CONCAT(' not in sql and 'CONCAT_WS(' not in sql
    with sqlite3.connect(':memory:') as con:
        con.execute('CREATE TABLE data (a TEXT, b TEXT)')
        con.executemany(
            'INSERT INTO data VALUES (?,?)',
            [('a', 'b'), (None, 'b'), ('a', None), (None, None), ('', 'b')],
        )
        assert con.execute(sql).fetchall() == [
            ('ab', 'a/b'), (None, 'b'), (None, 'a'), (None, ''), ('b', '/b'),
        ]


def test_explicit_rstrip_preserves_leading_whitespace_and_nbsp(compiler):
    t = ibis.table({'code': 'string'}, name='data')
    sql = compiler.compile(t.select(code=t.code.rstrip()))
    assert 'RTRIM(' in sql
    with sqlite3.connect(':memory:') as con:
        con.execute('CREATE TABLE data (code TEXT)')
        con.executemany(
            'INSERT INTO data VALUES (?)',
            [('  a   ',), ('b\t\n',), ('c\u00a0   ',), ('',), (None,)],
        )
        assert con.execute(sql).fetchall() == [
            ('  a',), ('b',), ('c\u00a0',), ('',), (None,),
        ]


def test_limit_uses_qualify_and_retains_sort(compiler, table):
    sql = compiler.compile(table.order_by(table.id.desc()).limit(2, offset=1))
    assert 'QUALIFY' in sql and 'ROW_NUMBER()' in sql
    assert 'TOP' not in sql and 'OFFSET' not in sql
    assert '"id" DESC' in sql


@pytest.mark.parametrize('build', [
    lambda t: t.label.split(','),
    lambda t: t.id.approx_nunique(),
    lambda t: t.ts.truncate('D'),
    lambda t: t.id.try_cast('int8'),
])
def test_unsupported_operations_fail_before_execution(compiler, table, build):
    with pytest.raises((exc.OperationNotDefinedError, exc.UnsupportedOperationError)):
        compiler.compile(build(table))


def test_projected_booleans_preserve_three_values(compiler):
    t = ibis.table({'id': 'int64', 'flag': 'boolean'}, name='data')
    sql = compiler.compile(
        t.select(
            t.id,
            flag=t.flag,
            negated=~t.flag,
            is_two=t.id == 2,
            both=t.flag & (t.id > 1),
        )
    )
    with sqlite3.connect(':memory:') as con:
        con.execute('CREATE TABLE data (id INTEGER, flag INTEGER)')
        con.executemany(
            'INSERT INTO data VALUES (?,?)', [(1, 1), (2, 0), (3, None), (None, None)]
        )
        assert con.execute(sql).fetchall() == [
            (1, 1, 0, 0, 0),
            (2, 0, 1, 1, 0),
            (3, None, None, 0, None),
            (None, None, None, None, None),
        ]


def test_null_safe_equality(compiler):
    t = ibis.table({'a': 'int64', 'b': 'int64'}, name='data')
    with sqlite3.connect(':memory:') as con:
        con.execute('CREATE TABLE data (a INTEGER, b INTEGER)')
        con.executemany(
            'INSERT INTO data VALUES (?,?)',
            [(None, None), (1, None), (None, 1), (1, 1), (1, 2)],
        )
        assert con.execute(
            compiler.compile(t.select(equal=t.a.identical_to(t.b)))
        ).fetchall() == [(1,), (0,), (0,), (1,), (0,)]


def test_join_preserves_duplicates_and_does_not_match_null_keys(compiler):
    remote = ibis.table({'id': 'int64', 'value': 'int64'}, name='remote')
    local = memtable({'id': [1, 1, None]}, schema={'id': 'int64'}, name='local')
    expr = remote.join(local, 'id').select(remote.id, remote.value)
    with sqlite3.connect(':memory:') as con:
        con.execute('CREATE TABLE remote (id INTEGER, value INTEGER)')
        con.execute('CREATE TABLE local (id INTEGER)')
        con.executemany('INSERT INTO remote VALUES (?,?)', [(1, 5), (2, 6), (None, 7)])
        con.executemany('INSERT INTO local VALUES (?)', [(1,), (1,), (None,)])
        assert con.execute(compiler.compile(expr)).fetchall() == [(1, 5), (1, 5)]
