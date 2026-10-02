from types import SimpleNamespace

import ibis
import ibis.common.exceptions as exc
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ibis_framework_teradata.datatypes import TeradataType, schema_from_cursor


@pytest.mark.parametrize('dtype,sql', [
    ('boolean', 'BYTEINT'),
    ('int8', 'BYTEINT'),
    ('int16', 'SMALLINT'),
    ('int32', 'INTEGER'),
    ('int64', 'BIGINT'),
    ('uint8', 'SMALLINT'),
    ('uint16', 'INTEGER'),
    ('uint32', 'BIGINT'),
    ('uint64', 'DECIMAL(20, 0)'),
    ('float32', 'FLOAT'),
    ('float64', 'FLOAT'),
    ('decimal(18, 2)', 'DECIMAL(18, 2)'),
    ('date', 'DATE'),
    ('time', 'TIME(6)'),
    ('timestamp', 'TIMESTAMP(6)'),
    ('timestamp(\'UTC\', 3)', 'TIMESTAMP(3) WITH TIME ZONE'),
])
def test_storage_types(dtype, sql):
    assert TeradataType.to_string(ibis.dtype(dtype)) == sql
    assert TeradataType.from_ibis(ibis.dtype(dtype)).sql('teradata') == ('INT' if sql == 'INTEGER' else sql)


@pytest.mark.parametrize('sql,dtype', [
    ('BYTEINT', 'int8'),
    ('SMALLINT', 'int16'),
    ('INTEGER', 'int32'),
    ('BIGINT', 'int64'),
    ('FLOAT', 'float64'),
    ('DECIMAL(18,2)', 'decimal(18,2)'),
    ('DATE', 'date'),
    ('TIMESTAMP(3)', 'timestamp(3)'),
    ('TIMESTAMP(6) WITH TIME ZONE', 'timestamp(\'UTC\',6)'),
    ('VARBYTE(10)', 'binary'),
    ('TIME(0)', 'time'),
])
def test_reflection_types(sql, dtype):
    assert TeradataType.from_string(sql) == ibis.dtype(dtype)


@pytest.mark.parametrize('dtype', [
    'array<int64>',
    'map<string,int64>',
    'json',
    'null',
    'timestamp(9)',
    'timestamp(\'Europe/London\')',
    'decimal(40,2)',
])
def test_unsupported_storage_types(dtype):
    with pytest.raises(exc.UnsupportedBackendType):
        TeradataType.to_string(ibis.dtype(dtype))


@given(st.integers(1, 38), st.integers(0, 38))
def test_decimal_roundtrip(precision, scale):
    if scale > precision:
        return
    dtype = ibis.dtype(f'decimal({precision},{scale})')
    assert TeradataType.from_string(TeradataType.to_string(dtype)) == dtype


def test_native_metadata_retains_width_scale_and_nullability():
    cursor = SimpleNamespace(
        description=[
            ('small', None, None, 1, None, None, False),
            ('amount', None, None, 8, 18, 2, True),
            ('ts', None, None, 8, None, 3, True),
        ],
        columntypename=['BYTEINT', 'DECIMAL', 'TIMESTAMP'],
    )
    assert schema_from_cursor(cursor) == ibis.schema(
        {'small': '!int8', 'amount': 'decimal(18,2)', 'ts': 'timestamp(3)'}
    )


def test_unknown_metadata_is_not_guessed():
    with pytest.raises(exc.UnsupportedBackendType):
        TeradataType.from_string('PERIOD(DATE)')


def test_duplicate_metadata_columns_rejected():
    cursor = SimpleNamespace(
        description=[('x', None, None, None, None, None, True)] * 2,
        columntypename=['INTEGER'] * 2,
    )
    with pytest.raises(exc.IbisError, match='Duplicate'):
        schema_from_cursor(cursor)
