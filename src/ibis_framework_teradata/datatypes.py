"""Teradata scalar storage types and DB-API result metadata."""

from __future__ import annotations

import re

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import sqlglot.expressions as sge
from ibis.backends.sql.datatypes import SqlglotType


class TeradataType(SqlglotType):
    dialect = 'teradata'
    default_decimal_precision = 38
    default_decimal_scale = 9
    default_temporal_scale = 6

    @classmethod
    def from_ibis(cls, dtype):
        text = cls.to_string(dtype)
        if dtype.is_int8() or dtype.is_boolean():
            # SQLGlot's Teradata parser maps BYTEINT to SMALLINT. Preserve it.
            return sge.DataType(this=sge.DataType.Type.USERDEFINED, kind='BYTEINT')
        return sge.DataType.build(text, dialect=cls.dialect)

    @classmethod
    def to_string(cls, dtype):
        if dtype.is_boolean() or dtype.is_int8():
            return 'BYTEINT'
        if dtype.is_int16() or dtype.is_uint8():
            return 'SMALLINT'
        if dtype.is_int32() or dtype.is_uint16():
            return 'INTEGER'
        if dtype.is_int64() or dtype.is_uint32():
            return 'BIGINT'
        if dtype.is_uint64():
            return 'DECIMAL(20, 0)'
        if dtype.is_floating():
            return 'FLOAT'
        if dtype.is_decimal():
            precision = dtype.precision or cls.default_decimal_precision
            scale = (
                dtype.scale if dtype.scale is not None else cls.default_decimal_scale
            )
            if not 1 <= precision <= 38 or not 0 <= scale <= precision:
                raise exc.UnsupportedBackendType(
                    'Teradata decimals require 0 <= scale <= precision <= 38'
                )
            return f'DECIMAL({precision}, {scale})'
        if dtype.is_string():
            length = dtype.length or 16000
            if not 1 <= length <= 32000:
                raise exc.UnsupportedBackendType(
                    'Teradata Unicode VARCHAR length must be between 1 and 32000'
                )
            return f'VARCHAR({length})'
        if dtype.is_binary():
            return 'VARBYTE(64000)'
        if dtype.is_date():
            return 'DATE'
        if dtype.is_time():
            return 'TIME(6)'
        if dtype.is_timestamp():
            scale = dtype.scale if dtype.scale is not None else 6
            if not 0 <= scale <= 6:
                raise exc.UnsupportedBackendType(
                    'Teradata timestamp precision must be between 0 and 6'
                )
            if dtype.timezone not in (None, 'UTC'):
                raise exc.UnsupportedBackendType(
                    'Use UTC for timezone-aware Teradata timestamps'
                )
            return f'TIMESTAMP({scale})' + (' WITH TIME ZONE' if dtype.timezone else '')
        raise exc.UnsupportedBackendType(
            f'Teradata does not support {dtype}; provide an explicit supported schema'
        )

    @classmethod
    def from_string(cls, text, nullable=None):
        nullable = cls.default_nullable if nullable is None else nullable
        name = re.sub(r'\s+', ' ', text.strip().upper())
        simple = {
            'BYTEINT': dt.Int8,
            'SMALLINT': dt.Int16,
            'INTEGER': dt.Int32,
            'INT': dt.Int32,
            'BIGINT': dt.Int64,
            'FLOAT': dt.Float64,
            'REAL': dt.Float64,
            'DOUBLE PRECISION': dt.Float64,
            'DATE': dt.Date,
            'BYTE': dt.Binary,
            'VARBYTE': dt.Binary,
            'BLOB': dt.Binary,
            'CHAR': dt.String,
            'VARCHAR': dt.String,
            'CLOB': dt.String,
        }
        base = name.split('(', 1)[0].strip()
        if base in simple:
            if base in ('CHAR', 'VARCHAR') and (
                match := re.fullmatch(r'(?:VAR)?CHAR\((\d+)\)', name)
            ):
                return dt.String(length=int(match[1]), nullable=nullable)
            return simple[base](nullable=nullable)
        if match := re.fullmatch(
            r'(?:DECIMAL|NUMERIC|NUMBER)(?:\((\d+)(?:,\s*(\d+))?\))?', name
        ):
            return dt.Decimal(
                int(match[1]) if match[1] else 38,
                int(match[2]) if match[2] else 0,
                nullable=nullable,
            )
        if match := re.fullmatch(
            r'(TIMESTAMP|TIME)(?:\((\d+)\))?( WITH TIME ZONE)?', name
        ):
            if match[1] == 'TIME':
                if match[3]:
                    raise exc.UnsupportedBackendType(
                        'Timezone-aware TIME is not supported'
                    )
                return dt.Time(nullable=nullable)
            return dt.Timestamp(
                timezone='UTC' if match[3] else None,
                scale=int(match[2]) if match[2] else 6,
                nullable=nullable,
            )
        raise exc.UnsupportedBackendType(
            f'Unsupported Teradata type {text!r}; pass an explicit schema to sql()'
        )


def schema_from_cursor(cursor):
    """Native names retain integer widths that DB-API NUMBER cannot represent."""
    description, names = cursor.description, cursor.columntypename
    if description is None or names is None or len(description) != len(names):
        raise exc.IbisError(
            'Teradata did not return metadata; pass an explicit schema to sql()'
        )
    fields, seen = [], set()
    for (name, _, _, _, precision, scale, nullable), type_name in zip(
        description, names
    ):
        if name in seen:
            raise exc.IbisError(
                f'Duplicate result column {name!r}; use distinct aliases'
            )
        seen.add(name)
        dtype = TeradataType.from_string(type_name, nullable=nullable)
        if dtype.is_decimal() and precision is not None:
            dtype = dt.Decimal(precision, scale or 0, nullable=dtype.nullable)
        elif dtype.is_timestamp() and scale is not None:
            dtype = dtype.copy(scale=scale)
        fields.append((name, dtype))
    return sch.Schema(dict(fields))
