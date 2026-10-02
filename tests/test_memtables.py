import datetime
from decimal import Decimal

import ibis
import ibis.common.exceptions as exc
import pyarrow as pa
import pytest

from conftest import memtable


def test_bounded_upload_preserves_duplicates_nulls_and_decimals(backend, connection):
    table = memtable({
        'id': [1, 1, None, 2, 3],
        'value': [Decimal('1.25'), Decimal('1.25'), None, Decimal('0.00'), Decimal('-2.50')],
        'flag': [True, False, None, True, False],
    },
        schema={'id': 'int64', 'value': 'decimal(18,2)', 'flag': 'boolean'},
        name='lookup',
    )
    backend._run_pre_execute_hooks(table)
    assert [len(rows) for _, rows in connection.batches] == [2, 2, 1]
    assert [row for _, rows in connection.batches for row in rows] == [
        [1, Decimal('1.25'), 1],
        [1, Decimal('1.25'), 0],
        [None, None, None],
        [2, Decimal('0.00'), 1],
        [3, Decimal('-2.50'), 0],
    ]
    ddl = connection.statements[0][0]
    assert 'CREATE MULTISET VOLATILE TABLE' in ddl
    assert 'NO PRIMARY INDEX ON COMMIT PRESERVE ROWS' in ddl
    assert 'VALUES (?, ?, ?)' in connection.batches[0][0]
    assert all(cursor.connection is connection for cursor in connection.cursors)


def test_repeated_execution_reuses_upload(backend, connection):
    table = memtable({'id': [1, 2]}, name='lookup')
    backend.execute(table)
    backend.execute(table)
    assert len(connection.batches) == 1
    assert sum(query.startswith('CREATE') for query, _, _ in connection.statements) == 1


def test_empty_typed_memtable_creates_table_without_inserting(backend, connection):
    table = memtable(pa.table({'id': pa.array([], type=pa.int64())}), name='empty')
    backend._run_pre_execute_hooks(table)
    assert connection.batches == []
    assert backend.get_schema('empty') == table.schema()


def test_arrow_path_never_converts_to_pandas(backend, connection, mocker):
    from ibis.formats.pyarrow import PyArrowTableProxy

    to_frame = mocker.patch.object(
        PyArrowTableProxy,
        'to_frame',
        side_effect=AssertionError('unnecessary pandas conversion'),
    )
    table = memtable(pa.table({'id': [1, 2]}))
    backend._run_pre_execute_hooks(table)
    to_frame.assert_not_called()


def test_polars_upload_and_result(backend, connection, mocker):
    pl = pytest.importorskip('polars')
    from ibis.formats.polars import PolarsDataFrameProxy

    to_frame = mocker.patch.object(
        PolarsDataFrameProxy,
        'to_frame',
        side_effect=AssertionError('unnecessary pandas conversion'),
    )
    local = memtable(
        pl.DataFrame({'id': [1, 1, None], 'label': ['é', '中文', None]}),
        schema={'id': 'int64', 'label': 'string'},
        name='polars_lookup',
    )
    backend._run_pre_execute_hooks(local)
    assert [row for _, rows in connection.batches for row in rows] == [
        [1, 'é'], [1, '中文'], [None, None],
    ]
    to_frame.assert_not_called()
    connection.responses['"remote"'] = ([(1,)], None, None)
    result = backend.to_polars(
        backend.sql('SELECT id FROM "remote"', schema={'id': 'int64'})
    )
    assert result.to_dict(as_series=False) == {'id': [1]}


def test_unicode_and_binary_widths_follow_actual_data(backend, connection):
    table = memtable(
        {'text': ['a', '中文', '😀'], 'blob': [b'', b'abc', None]},
        schema={'text': 'string', 'blob': 'binary'},
        name='data',
    )
    backend._run_pre_execute_hooks(table)
    ddl = connection.statements[0][0]
    assert 'VARCHAR(2) CHARACTER SET UNICODE CASESPECIFIC' in ddl
    assert 'VARBYTE(3)' in ddl


def test_null_typed_column_fails_without_creating_table(backend, connection):
    table = memtable(pa.table({'x': pa.nulls(2)}))
    with pytest.raises(exc.UnsupportedBackendType, match='explicit supported schema'):
        backend._run_pre_execute_hooks(table)
    assert connection.statements == []


def test_upload_failure_drops_partial_table_and_can_retry(backend, connection):
    table = memtable({'id': [1, 2]}, name='retry')
    connection.fail_batch = True
    with pytest.raises(RuntimeError, match='upload'):
        backend._run_pre_execute_hooks(table)
    assert connection.statements[-1][0] == 'DROP TABLE "retry"'
    assert not backend._staged_names and table.op() not in backend._memtables
    connection.fail_batch = False
    backend._run_pre_execute_hooks(table)
    assert 'retry' in backend._staged_names


def test_failed_create_never_drops_a_preexisting_table(backend, connection):
    connection.fail = lambda query: query.startswith('CREATE')
    with pytest.raises(RuntimeError):
        backend._run_pre_execute_hooks(memtable({'id': [1]}, name='existing'))
    assert not any(query.startswith('DROP') for query, _, _ in connection.statements)


def test_failed_cleanup_closes_poisoned_session(backend, connection):
    connection.fail_batch = True
    connection.fail = lambda query: query.startswith('DROP')
    with pytest.raises(RuntimeError, match='upload'):
        backend._run_pre_execute_hooks(memtable({'id': [1]}))
    assert connection.closed and backend.con is None
    assert not backend._memtables and not backend._temp_schemas


def test_cleanup_allows_reupload(backend, connection):
    table = memtable({'id': [1]}, name='lookup')
    backend._run_pre_execute_hooks(table)
    backend.cleanup_memtables()
    assert not backend._staged_names and not backend._temp_schemas
    backend._run_pre_execute_hooks(table)
    assert len(connection.batches) == 2


def test_new_operation_cannot_reuse_an_uploaded_name(backend):
    first = memtable({'id': [1]}, name='same')
    second = memtable({'id': [2]}, name='same')
    backend._run_pre_execute_hooks(first)
    with pytest.raises(exc.IbisError, match='different staged'):
        backend._run_pre_execute_hooks(second)


@pytest.mark.parametrize('value', [
    float('nan'), float('inf'), float('-inf'),
])
def test_non_finite_values_are_not_silently_changed(backend, connection, value):
    table = memtable(pa.table({'x': [value]}), name='floats')
    with pytest.raises(ValueError, match='NaN or infinity'):
        backend._run_pre_execute_hooks(table)
    assert connection.statements[-1][0] == 'DROP TABLE "floats"'


def test_unsupported_expression_does_not_upload_memtable(backend, connection):
    table = memtable({'text': ['a,b']})
    with pytest.raises(exc.OperationNotDefinedError):
        backend.execute(table.text.split(','))
    assert connection.statements == []


def test_unsupported_interval_precision_does_not_upload_memtable(backend, connection):
    table = memtable({'n': [1]})
    with pytest.raises(exc.UnsupportedOperationError, match='microsecond'):
        backend.execute(table.n.as_interval('ns'))
    assert connection.statements == []


def test_reconnect_invalidates_all_registration_state(mocker, connection):
    from conftest import FakeConnection

    from ibis_framework_teradata import Backend

    replacement = FakeConnection()
    mocker.patch(
        'ibis_framework_teradata.teradatasql.connect',
        side_effect=[connection, replacement],
    )
    backend = Backend().connect(host='td')
    table = memtable({'id': [1]}, name='lookup')
    backend._run_pre_execute_hooks(table)
    backend.reconnect()
    assert connection.closed and backend.con is replacement
    assert not backend._staged_names and not backend._memtables
    backend._run_pre_execute_hooks(table)
    assert len(replacement.batches) == 1


def test_utc_timestamps_retain_timezone(backend):
    value = datetime.datetime(
        2026, 1, 1, 12,
        tzinfo=datetime.timezone(datetime.timedelta(hours=2))
    )
    result = backend._bind_value(value, ibis.dtype('timestamp(\'UTC\')'))
    assert result == datetime.datetime(2026, 1, 1, 10, tzinfo=datetime.UTC)


def test_uint64_maximum_binds_as_exact_decimal(backend, connection):
    table = memtable(pa.table({'id': pa.array([2 ** 64 - 1], type=pa.uint64())}))
    backend._run_pre_execute_hooks(table)
    value = connection.batches[0][1][0][0]
    assert isinstance(value, Decimal) and value == Decimal(2 ** 64 - 1)
