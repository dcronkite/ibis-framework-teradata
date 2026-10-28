"""An installable Ibis backend for Teradata.

Installing this distribution registers ``ibis.teradata``. The ETL application
uses normal Ibis APIs; importing this module is not necessary.
"""

from __future__ import annotations

import contextlib
import datetime
import math
import weakref
from urllib.parse import parse_qsl, unquote

import ibis
import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.types as ir
import sqlglot as sg
import sqlglot.expressions as sge
import teradatasql
from ibis.backends import HasCurrentDatabase
from ibis.backends.sql import SQLBackend

from ibis_framework_teradata._version import __version__
from ibis_framework_teradata.compiler import compiler
from ibis_framework_teradata.datatypes import TeradataType, schema_from_cursor

__all__ = ['Backend', '__version__']


class Backend(SQLBackend, HasCurrentDatabase):
    name = 'teradata'
    compiler = compiler
    supports_temporary_tables = True
    supports_create_or_replace = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.con = None
        self._owns_connection = True
        self._temp_schemas = {}
        self._staged_names = set()
        self._batch_size = 10000

    def do_connect(self, host, user=None, password=None, database=None, *, batch_size=10000, **kwargs):
        """Open one session, reused for uploading memtables and executing SQL.

        Other kwargs are passed to teradatasql.connect, including ``logmech``,
        ``encryptdata``, ``dbs_port`` and ``tmode``. Driver options use the
        driver's documented names and value types. Managed staging requires
        autocommit; explicit multi-statement transactions are not supported.
        """
        self._validate_batch_size(batch_size)
        if kwargs.pop('autocommit', True) is not True:
            raise exc.UnsupportedOperationError(
                'Teradata memtable staging requires autocommit=True'
            )
        # Closing the old session also removes its volatile tables.
        if self.con is not None:
            self.disconnect()
        options = {'host': host, 'tmode': 'ANSI', **kwargs}
        for key, value in (('user', user), ('password', password), ('database', database)):
            if value is not None:
                options[key] = value
        con = teradatasql.connect(**options)
        try:
            con.autocommit = True
        except Exception:
            con.close()
            raise
        self.con = con
        self._owns_connection = True
        self._batch_size = batch_size
        self._clear_session_state()

    @staticmethod
    def _validate_batch_size(size):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('batch_size must be a positive integer')

    @classmethod
    def from_connection(cls, con, /, *, batch_size=10000, close_on_disconnect=False):
        """Wrap a driver's autocommit connection; preserve caller ownership by default."""
        cls._validate_batch_size(batch_size)
        if con.autocommit is not True:
            raise exc.UnsupportedOperationError(
                'Teradata memtable staging requires an autocommit connection'
            )
        backend = cls(con)
        backend.con = con
        backend._can_reconnect = False
        backend._owns_connection = close_on_disconnect
        backend._batch_size = batch_size
        return backend

    def _from_url(self, url, **overrides):
        options = dict(parse_qsl(url.query))
        if url.hostname:
            options['host'] = url.hostname
        if url.username is not None:
            options['user'] = unquote(url.username)
        if url.password is not None:
            options['password'] = unquote(url.password)
        if url.path.strip('/'):
            options['database'] = unquote(url.path.strip('/'))
        if url.port is not None:
            options['dbs_port'] = str(url.port)
        options.update(overrides)
        if 'batch_size' in options:
            options['batch_size'] = int(options['batch_size'])
        return self.connect(**options)

    def _require_connection(self):
        if self.con is None:
            raise exc.IbisError('Teradata backend is disconnected')
        return self.con

    def _clear_session_state(self):
        self._memtables = weakref.WeakSet()
        self._temp_schemas.clear()
        self._staged_names.clear()
        self.__dict__.pop('version', None)

    def disconnect(self):
        con = self.con
        if con is None:
            return
        try:
            if self._owns_connection:
                con.close()
            else:
                # Volatile tables persist while a caller-owned connection lives.
                for name in tuple(self._temp_schemas):
                    self.drop_table(name, force=True)
        finally:
            self.con = None
            self._clear_session_state()

    def __enter__(self):
        self._require_connection()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def raw_sql(self, query, /, params=None, **kwargs):
        """Execute raw SQL. The caller must close the returned cursor."""
        if isinstance(query, sge.Expression):
            query = query.sql(self.dialect)
        cursor = self._require_connection().cursor()
        try:
            if params is None:
                cursor.execute(query, **kwargs)
            else:
                cursor.execute(query, params, **kwargs)
        except Exception:
            cursor.close()
            raise
        return cursor

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        cursor = self.raw_sql(*args, **kwargs)
        try:
            yield cursor
        finally:
            cursor.close()

    @property
    def current_database(self):
        with self._safe_raw_sql('SELECT DATABASE') as cursor:
            return cursor.fetchone()[0].strip()

    @property
    def version(self):
        with self._safe_raw_sql('SELECT InfoData FROM DBC.DBCInfoV WHERE InfoKey = \'VERSION\'') as cursor:
            return cursor.fetchone()[0].strip()

    def list_databases(self, like=None):
        with self._safe_raw_sql('SELECT DatabaseName FROM DBC.DatabasesV') as cursor:
            values = [row[0].strip() for row in cursor.fetchall()]
        return sorted(self._filter_with_like(values, like))

    @staticmethod
    def _validate_database(database):
        if database is not None and not isinstance(database, str):
            raise ValueError('Teradata uses one database name; catalogs and database tuples are not supported')

    def _table_sql(self, name, database=None):
        self._validate_database(database)
        return sg.table(name, db=database, quoted=True).sql(self.dialect)

    def list_tables(self, like=None, database=None):
        self._validate_database(database)
        with self._safe_raw_sql(
                'SELECT TableName FROM DBC.TablesV WHERE DatabaseName = ? AND TableKind IN (\'T\', \'V\', \'O\')',
                params=[database or self.current_database],
        ) as cursor:
            values = [row[0].strip() for row in cursor.fetchall()]
        # volatile tables do not appear in DBC.TablesV
        if database is None:
            values.extend(self._temp_schemas)
        return sorted(self._filter_with_like(set(values), like))

    def table(self, name, /, *, database=None):
        self._validate_database(database)
        schema = self.get_schema(name, database=database)
        return ops.DatabaseTable(name, schema=schema, source=self, namespace=ops.Namespace(database=database)).to_expr()

    def get_schema(self, name, *, catalog=None, database=None):
        if catalog is not None:
            raise ValueError('Teradata does not support a catalog namespace')
        self._validate_database(database)
        if database is None and name in self._temp_schemas:
            return self._temp_schemas[name]
        # avoids dictionary-access requirements and preserves BYTEINT, DECIMAL precision, and timestamp
        with self._safe_raw_sql(f'SELECT * FROM {self._table_sql(name, database)} WHERE 1 = 0') as cursor:
            return schema_from_cursor(cursor)

    def _get_schema_using_query(self, query):
        parsed = sg.parse_one(query, read=self.dialect)
        if not isinstance(parsed, (sge.Select, sge.SetOperation)):
            raise TypeError('sql() requires a SELECT query')
        source = parsed.subquery('ibis_schema', copy=True)
        sql = sg.select('*').from_(source).where('1 = 0').sql(self.dialect)
        with self._safe_raw_sql(sql) as cursor:
            return schema_from_cursor(cursor)

    def _column_definitions(self, schema, arrow=None):
        fields = []
        for name, dtype in schema.items():
            text = TeradataType.to_string(dtype)
            if arrow is not None and (dtype.is_string() or dtype.is_binary()):
                if dtype.is_string():
                    width = max(
                        (len(value.as_py().encode('utf-16-le')) // 2 for value in arrow[name] if value.is_valid),
                        default=1,
                    )
                    if dtype.length is not None:
                        if width > dtype.length:
                            raise ValueError(f'Values in {name!r} exceed the declared string length')
                        width = dtype.length
                    if width > 32000:
                        raise exc.UnsupportedBackendType(f'Values in {name!r} exceed the Unicode VARCHAR limit')
                    text = f'VARCHAR({max(1, width)})'
                else:
                    width = max((len(value.as_py()) for value in arrow[name] if value.is_valid), default=1)
                    if width > 64000:
                        raise exc.UnsupportedBackendType(f'Values in {name!r} exceed the VARBYTE limit')
                    text = f'VARBYTE({max(1, width)})'
            if dtype.is_string():
                text += ' CHARACTER SET UNICODE CASESPECIFIC'
            null = '' if dtype.nullable else ' NOT NULL'
            fields.append(f'{sg.to_identifier(name, quoted=True).sql(self.dialect)} {text}{null}')
        if not fields:
            raise ValueError('Teradata tables require at least one column')
        return ', '.join(fields)

    def _create_sql(self, name, schema, *, database=None, temp=False, arrow=None):
        if temp and database is not None:
            raise ValueError('Volatile tables cannot specify a database')
        kind = 'MULTISET VOLATILE' if temp else 'MULTISET'
        suffix = ' ON COMMIT PRESERVE ROWS' if temp else ''
        return f'CREATE {kind} TABLE {self._table_sql(name, database)} ({self._column_definitions(schema, arrow)}) NO PRIMARY INDEX{suffix}'

    @staticmethod
    def _bind_value(value, dtype):
        if value is None:
            if not dtype.nullable:
                raise ValueError('NULL value for a non-nullable memtable column')
            return None
        if dtype.is_boolean():
            return int(value)
        if dtype.is_uint64():
            from decimal import Decimal

            return Decimal(value)
        if dtype.is_floating() and not math.isfinite(value):
            raise ValueError('Teradata memtables cannot contain NaN or infinity; replace them explicitly before upload')
        if dtype.is_timestamp() and value.tzinfo is not None:
            return value.astimezone(datetime.UTC)
        return value

    def _insert_arrow(self, cursor, name, schema, arrow):
        columns = ', '.join(sg.to_identifier(n, quoted=True).sql(self.dialect) for n in schema.names)
        markers = ', '.join('?' for _ in schema.names)
        query = f'INSERT INTO {self._table_sql(name)} ({columns}) VALUES ({markers})'
        for batch in arrow.to_batches(max_chunksize=self._batch_size):
            # Materialize only one bounded batch, retaining Decimal and NULLs.
            arrays = [column.to_pylist() for column in batch.columns]
            rows = [
                [self._bind_value(v, dtype) for v, dtype in zip(row, schema.types)]
                for row in zip(*arrays)
            ]
            if rows:
                cursor.executemany(query, rows)

    @staticmethod
    def _memtable_to_arrow(op):
        # Arrow handles nullable decimals without changing the source dataframe or converting exact values to floats
        from ibis.formats.pandas import PandasDataFrameProxy

        if isinstance(op.data, PandasDataFrameProxy):
            import pyarrow as pa

            return pa.Table.from_pandas(op.data.obj, schema=op.schema.to_pyarrow(), preserve_index=False, safe=True)
        return op.data.to_pyarrow(op.schema)

    def _discard_created_table(self, name, database=None):
        try:
            self.drop_table(name, database=database, force=True)
        except Exception:  # noqa: BLE001 -- preserve the original upload error
            # never permit a session containing partially loaded tables to be reused,
            # even if closing that broken connection also fails
            broken, self.con = self.con, None
            self._clear_session_state()
            with contextlib.suppress(Exception):
                broken.close()

    def _register_in_memory_table(self, op):
        self._require_connection()
        if op.name in self._temp_schemas:
            raise exc.IbisError(f'A different staged table already uses {op.name!r}; use a new memtable name')
        arrow = self._memtable_to_arrow(op)
        query = self._create_sql(op.name, op.schema, temp=True, arrow=arrow)
        created = False
        try:
            with self._safe_raw_sql(query) as cursor:
                created = True
                self._insert_arrow(cursor, op.name, op.schema, arrow)
        except Exception:
            if created:
                # drop only tables that this upload actually created
                self._discard_created_table(op.name)
            raise
        self._temp_schemas[op.name] = op.schema
        self._staged_names.add(op.name)

    def _make_memtable_finalizer(self, name):
        # explicit job/session cleanup avoids retaining connections via atexit
        return None

    def cleanup_memtables(self):
        """Drop uploaded memtables. Reusing an expression uploads it again."""
        for name in tuple(self._staged_names):
            self.drop_table(name, force=True)
        self._memtables = weakref.WeakSet()

    def drop_table(self, name, /, *, database=None, force=False):
        kwargs = {'ignoreErrors': [3807]} if force else {}
        with self._safe_raw_sql(f'DROP TABLE {self._table_sql(name, database)}', **kwargs):
            pass
        if database is None:
            self._temp_schemas.pop(name, None)
            self._staged_names.discard(name)
            self._memtables = weakref.WeakSet(op for op in self._memtables if op.name != name)

    def truncate_table(self, name, /, *, database=None):
        with self._safe_raw_sql(f'DELETE FROM {self._table_sql(name, database)}'):
            pass

    def create_table(self, name, /, obj=None, *, schema=None,
                     database=None, temp=False, overwrite=False):
        if obj is None and schema is None:
            raise ValueError('Specify obj or schema')
        source = obj if isinstance(obj, ir.Table) else ibis.memtable(obj, schema=schema) if obj is not None else None
        schema = ibis.schema(schema) if schema is not None else source.schema()
        if source is not None and source.schema() != schema:
            source = source.cast(schema)
        arrow = (
            self._memtable_to_arrow(source.op())
            if source is not None and isinstance(source.op(), ops.InMemoryTable)
            else None
        )
        create = self._create_sql(name, schema, database=database, temp=temp, arrow=arrow)
        if source is not None:
            sql = self.compile(source)
            self._run_pre_execute_hooks(source)
        if overwrite:
            self.drop_table(name, database=database, force=True)
        with self._safe_raw_sql(create):
            pass
        try:
            if source is not None:
                columns = ', '.join(sg.to_identifier(n, quoted=True).sql(self.dialect) for n in schema.names)
                with self._safe_raw_sql(f'INSERT INTO {self._table_sql(name, database)} ({columns}) {sql}'):
                    pass
        except Exception:
            self._discard_created_table(name, database)
            raise
        if temp:
            self._temp_schemas[name] = schema
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def execute(self, expr, /, *, params=None, limit=None, **kwargs):
        # compile before staging so unsupported expressions cannot trigger ddl
        table = expr.as_table()
        sql = self.compile(table, params=params, limit=limit, **kwargs)
        self._run_pre_execute_hooks(expr)
        with self._safe_raw_sql(sql) as cursor:
            result = self._fetch_from_cursor(cursor, table.schema())
        return expr.__pandas_result__(result)

    def _fetch_from_cursor(self, cursor, schema):
        import pandas as pd
        from ibis.formats.pandas import PandasData

        # inferring dtype with nullable Python integers first results in float, so use object to avoid losing data
        frame = pd.DataFrame(cursor.fetchall(), columns=schema.names, dtype=object)
        return PandasData.convert_table(frame, schema)

    @staticmethod
    def _arrow_batch(rows, schema):
        import pyarrow as pa

        columns = list(zip(*rows))
        arrays = []
        for values, dtype, field in zip(columns, schema.types, schema.to_pyarrow()):
            if dtype.is_boolean():
                values = [None if v is None else bool(v) for v in values]
            arrays.append(pa.array(values, type=field.type))
        return pa.RecordBatch.from_arrays(arrays, schema=schema.to_pyarrow())

    def to_pyarrow_batches(self, expr, /, *, params=None,
                           limit=None, chunk_size=1000000, **kwargs):
        import pyarrow as pa

        self._validate_batch_size(chunk_size)
        sql = self.compile(expr, params=params, limit=limit)
        self._run_pre_execute_hooks(expr)
        schema = expr.as_table().schema()
        cursor = self.raw_sql(sql)

        def batches():
            try:
                while rows := cursor.fetchmany(chunk_size):
                    yield self._arrow_batch(rows, schema)
            finally:
                cursor.close()

        try:
            reader = pa.RecordBatchReader.from_batches(schema.to_pyarrow(), batches())
        except Exception:
            cursor.close()
            raise
        # release the cursor if an unconsumed reader is garbage collected
        weakref.finalize(reader, cursor.close)
        return reader
