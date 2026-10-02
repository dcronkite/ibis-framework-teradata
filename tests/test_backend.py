from decimal import Decimal
from urllib.parse import urlparse

import ibis
import ibis.common.exceptions as exc
import pytest

from conftest import memtable
from ibis_framework_teradata import Backend


def test_connect_forwards_driver_options_and_uses_one_session(mocker, connection):
    connect = mocker.patch('ibis_framework_teradata.teradatasql.connect', return_value=connection)
    backend = Backend().connect(host='td', user='u', password='p', database='db', encryptdata='true', batch_size=2)
    assert backend.con is connection
    connect.assert_called_once_with(host='td', user='u', password='p', database='db', tmode='ANSI', encryptdata='true')
    assert connection.autocommit is True


def test_url_decodes_credentials_and_port(mocker, connection):
    connect = mocker.patch('ibis_framework_teradata.teradatasql.connect', return_value=connection)
    backend = Backend()._from_url(
        urlparse('teradata://some%40user:a%2Bb%2F@td:1026/analytics?logmech=LDAP&batch_size=2')
    )
    assert backend._batch_size == 2
    assert connect.call_args.kwargs == {
        'host': 'td',
        'user': 'some@user',
        'password': 'a+b/',
        'database': 'analytics',
        'dbs_port': '1026',
        'logmech': 'LDAP',
        'tmode': 'ANSI',
    }


def test_raw_sql_cursor_ownership_and_error_cleanup(backend, connection):
    cursor = backend.raw_sql('SELECT 1')
    assert not cursor.closed
    cursor.close()
    with backend._safe_raw_sql('SELECT 2') as cursor:
        assert not cursor.closed
    assert cursor.closed
    connection.fail = lambda query: True
    with pytest.raises(RuntimeError):
        backend.raw_sql('broken')
    assert connection.cursors[-1].closed


def test_native_table_reflection_and_namespace(backend, connection):
    connection.responses['"analytics"."sales"'] = (
        [],
        [('id', None, None, 1, None, None, False), ('value', None, None, 8, 18, 2, True)],
        ['BYTEINT', 'DECIMAL'],
    )
    table = backend.table('sales', database='analytics')
    assert table.schema() == ibis.schema({'id': '!int8', 'value': 'decimal(18,2)'})
    assert '"analytics"."sales"' in backend.compile(table)
    assert 'WHERE 1 = 0' in connection.statements[-1][0]


def test_explicit_sql_schema_needs_no_metadata_query(backend, connection):
    table = backend.sql('SELECT 1 AS x', schema={'x': 'int64'})
    assert table.schema() == ibis.schema({'x': 'int64'})
    assert connection.statements == []


def test_query_schema_zero_row_wrapper(backend, connection):
    connection.responses['ibis_schema'] = (
        [],
        [('x', None, None, 8, None, None, True)],
        ['BIGINT'],
    )
    table = backend.sql('SELECT 1 AS x')
    assert table.schema() == ibis.schema({'x': 'int64'})
    assert 'WHERE 1 = 0' in connection.statements[-1][0]


def test_query_schema_wrapper_preserves_interval_precision(backend, connection):
    connection.responses['ibis_schema'] = (
        [],
        [('dt', None, None, 4, None, None, True)],
        ['DATE'],
    )
    table = backend.sql(
        'SELECT index_date - CAST(365 AS INTERVAL DAY(4)) AS dt FROM events'
    )
    assert table.schema() == ibis.schema({'dt': 'date'})
    assert 'CAST(365 AS INTERVAL DAY(4))' in connection.statements[-1][0]
    assert 'WHERE 1 = 0' in connection.statements[-1][0]


def test_identifiers_are_quoted(backend):
    query = backend._create_sql(
        'odd"name', ibis.schema({'select"x': 'string'}), temp=True
    )
    assert '"odd""name"' in query and '"select""x"' in query
    assert 'CHARACTER SET UNICODE CASESPECIFIC' in query


@pytest.mark.parametrize('database', [('catalog', 'db'), ['db']])
def test_multi_level_namespaces_rejected(backend, database):
    with pytest.raises(ValueError):
        backend.table('x', database=database)


def test_create_volatile_table_and_cached_schema(backend, connection):
    table = backend.create_table(
        'temp', schema={'id': 'int64', 'active': 'boolean'}, temp=True
    )
    assert table.schema() == backend.table('temp').schema()
    assert len(connection.statements) == 1
    assert 'MULTISET VOLATILE' in connection.statements[0][0]
    assert 'ON COMMIT PRESERVE ROWS' in connection.statements[0][0]


def test_create_table_from_memtable_inserts_on_server(backend, connection):
    source = memtable({'id': [1, 1]}, name='source')
    result = backend.create_table('destination', source, temp=True)
    assert result.schema() == source.schema()
    assert any(query.startswith('INSERT INTO "destination"') for query, _, _ in connection.statements)
    assert len(connection.batches) == 1


def test_create_requires_data_or_schema(backend):
    with pytest.raises(ValueError):
        backend.create_table('x')
    with pytest.raises(ValueError):
        backend.create_table('x', schema={'id': 'int64'}, temp=True, database='db')


def test_force_drop_ignores_only_missing_table(backend, connection):
    backend.drop_table('unknown', force=True)
    assert connection.statements[-1][2] == {'ignoreErrors': [3807]}


def test_execute_preserves_decimal_values(backend, connection):
    connection.responses['"remote"'] = ([(Decimal('12.34'),), (None,)], None, None)
    expr = backend.sql('SELECT amount FROM "remote"', schema={'amount': 'decimal(18,2)'})
    result = backend.execute(expr)
    assert result.iloc[0, 0] == Decimal('12.34')
    assert isinstance(result.iloc[0, 0], Decimal)


def test_execute_preserves_nullable_large_integer_ids(backend, connection):
    connection.responses['"remote"'] = ([(2 ** 63 - 1,), (None,)], None, None)
    expr = backend.sql('SELECT id FROM "remote"', schema={'id': 'int64'})
    result = backend.execute(expr)
    assert result.iloc[0, 0] == 2 ** 63 - 1
    assert result.iloc[1, 0] is None


@pytest.mark.parametrize('output', ['pandas', 'arrow', 'polars'])
def test_results_preserve_char_padding_and_varchar_whitespace(backend, connection, output):
    rows = [
        ('  A   ', '  A   '),
        ('B\u00a0   ', 'B\u00a0'),
        ('\u00a0C  ', '\u00a0C  '),
        ('      ', ''),
        (None, None),
    ]
    connection.responses['"remote"'] = (rows, None, ['CHAR', 'VARCHAR'])
    expr = backend.sql(
        'SELECT fixed, varying FROM "remote"',
        schema={'fixed': 'string', 'varying': 'string'},
    )
    if output == 'pandas':
        result = backend.execute(expr)
        assert result.isna().iloc[-1].all()
        # pandas uses its native missing-string representation rather than None.
        actual = list(result.iloc[:-1].itertuples(index=False, name=None))
        actual.append((None, None))
    elif output == 'arrow':
        actual = [(row['fixed'], row['varying']) for row in backend.to_pyarrow(expr).to_pylist()]
    else:
        pytest.importorskip('polars')
        actual = backend.to_polars(expr).rows()
    assert actual == rows
    assert connection.cursors[-1].closed


def test_arrow_results_preserve_nullable_boolean_and_decimal(backend, connection):
    connection.responses['"remote"'] = (
        [(1, Decimal('1.25')), (None, None), (0, Decimal('2.50'))],
        None,
        None,
    )
    expr = backend.sql(
        'SELECT * FROM "remote"',
        schema={'active': 'boolean', 'amount': 'decimal(18,2)'},
    )
    result = backend.to_pyarrow(expr)
    assert result.to_pylist() == [
        {'active': True, 'amount': Decimal('1.25')},
        {'active': None, 'amount': None},
        {'active': False, 'amount': Decimal('2.50')},
    ]
    assert connection.cursors[-1].closed


def test_empty_arrow_results_retain_schema(backend, connection):
    expr = backend.sql('SELECT 1 WHERE 1=0', schema={'x': 'int64'})
    result = backend.to_pyarrow(expr)
    assert len(result) == 0 and result.column_names == ['x']
    assert connection.cursors[-1].closed


def test_disconnect_preserves_external_connection(backend, connection):
    backend.disconnect()
    assert not connection.closed
    with pytest.raises(exc.IbisError, match='disconnected'):
        backend.raw_sql('SELECT 1')


def test_context_closes_owned_connection(connection):
    with Backend.from_connection(connection, close_on_disconnect=True) as backend:
        assert backend.con is connection
    assert connection.closed


def test_caller_connection_cannot_reconnect(backend):
    with pytest.raises(exc.IbisError):
        backend.reconnect()


@pytest.mark.parametrize('size', [0, -1, True, 1.5, '2'])
def test_invalid_batch_size_rejected(connection, size):
    with pytest.raises(ValueError):
        Backend.from_connection(connection, batch_size=size)


def test_manual_transactions_rejected(connection):
    connection.autocommit = False
    with pytest.raises(exc.UnsupportedOperationError, match='autocommit'):
        Backend.from_connection(connection)


def test_independent_existing_connections_have_independent_identity():
    from conftest import FakeConnection

    assert Backend.from_connection(FakeConnection()) != Backend.from_connection(
        FakeConnection()
    )


def test_disconnect_cleans_created_tables_on_external_session(backend, connection):
    backend.create_table('temp', schema={'id': 'int64'}, temp=True)
    backend.disconnect()
    assert connection.statements[-1][0] == 'DROP TABLE "temp"'
    assert not connection.closed


def test_create_from_local_data_uses_observed_string_widths(backend, connection):
    backend.create_table('destination', {'text': ['a', 'é']}, temp=True)
    ddl = next(
        query for query, _, _ in connection.statements
        if query.startswith('CREATE MULTISET VOLATILE TABLE "destination"')
    )
    assert 'VARCHAR(1)' in ddl
