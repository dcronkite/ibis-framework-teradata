"""Compile the portable ETL subset of Ibis to Teradata SQL."""

from __future__ import annotations

import math
from decimal import Decimal
from functools import reduce
from typing import ClassVar

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import sqlglot as sg
import sqlglot.expressions as sge
from ibis.backends.sql.compilers.base import ALL_OPERATIONS, NULL, STAR, SQLGlotCompiler
from ibis.backends.sql.rewrites import (
    exclude_unsupported_window_frame_from_rank,
    exclude_unsupported_window_frame_from_row_number,
)
from sqlglot.dialects.teradata import Teradata
from sqlglot.tokens import TokenType

from ibis_framework_teradata.datatypes import TeradataType


class TeradataDialect(Teradata):
    class Parser(Teradata.Parser):
        def _parse_interval_type_unit(self):
            unit = self._parse_var(upper=True)
            if self._curr.token_type == TokenType.L_PAREN:
                return sge.Anonymous(
                    this=unit.name,
                    expressions=self._parse_wrapped_csv(self._parse_number),
                )
            return unit

        def _parse_types(
            self,
            check_func=False,
            schema=False,
            allow_identifiers=True,
            with_collation=False,
        ):
            # SQLGlot 30.21 does not consume precision after interval units.
            if (
                self._curr.token_type == TokenType.INTERVAL
                and self._next.text.upper()
                in ('YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND')
            ):
                self._advance()
                unit = self._parse_interval_type_unit()
                if self._match_text_seq('TO'):
                    unit = sge.IntervalSpan(
                        this=unit, expression=self._parse_interval_type_unit()
                    )
                return self.expression(
                    sge.DataType(this=sge.DataType.Type.INTERVAL, expressions=[unit])
                )
            return super()._parse_types(
                check_func=check_func,
                schema=schema,
                allow_identifiers=allow_identifiers,
                with_collation=with_collation,
            )

    class Generator(Teradata.Generator):
        def interval_sql(self, expression):
            # A calendar quarter is three months, rather than SQLGlot's 90 days.
            if expression.text('unit').upper().startswith('QUARTER'):
                inner = sge.Interval(
                    this=expression.this.copy(), unit=sge.Var(this='MONTH')
                )
                return f'(3 * {super().interval_sql(inner)})'
            return super().interval_sql(expression)


_SUPPORTED_VALUES = frozenset(
    [
        'Literal',
        'Field',
        'Alias',
        'Cast',
        'ScalarParameter',
        'ScalarSubquery',
        'ExistsSubquery',
        'InSubquery',
        'InValues',
        'Between',
        'Equals',
        'NotEquals',
        'Greater',
        'GreaterEqual',
        'Less',
        'LessEqual',
        'IdenticalTo',
        'And',
        'Or',
        'Xor',
        'Not',
        'IsNull',
        'NotNull',
        'IfElse',
        'SimpleCase',
        'SearchedCase',
        'Coalesce',
        'NullIf',
        'Add',
        'Subtract',
        'Multiply',
        'Divide',
        'FloorDivide',
        'Modulus',
        'Power',
        'Negate',
        'Abs',
        'Ceil',
        'Floor',
        'Round',
        'Sign',
        'Sqrt',
        'Exp',
        'Ln',
        'Log',
        'Log2',
        'Log10',
        'Lowercase',
        'Uppercase',
        'StringLength',
        'Substring',
        'StringSlice',
        'StringConcat',
        'StringJoin',
        'StringContains',
        'StartsWith',
        'EndsWith',
        'StringFind',
        'StringReplace',
        'Strip',
        'LStrip',
        'RStrip',
        'StringSQLLike',
        'StringSQLILike',
        'Count',
        'CountStar',
        'CountDistinct',
        'Sum',
        'Mean',
        'Min',
        'Max',
        'Any',
        'All',
        'Variance',
        'StandardDev',
        'RowNumber',
        'MinRank',
        'DenseRank',
        'Lag',
        'Lead',
        'FirstValue',
        'LastValue',
        'First',
        'Last',
        'WindowFunction',
        'WindowBoundary',
        'SortKey',
        'Date',
        'TimestampNow',
        'DateNow',
        'ExtractYear',
        'ExtractMonth',
        'ExtractDay',
        'ExtractHour',
        'ExtractMinute',
        'ExtractSecond',
        'ExtractQuarter',
        'DateAdd',
        'DateSub',
        'TimestampAdd',
        'TimestampSub',
        'IntervalAdd',
        'IntervalSubtract',
        'IntervalMultiply',
        'IntervalFromInteger',
        'DateDiff',
    ]
)
_STORED_BOOLEANS = frozenset(
    [
        'Literal',
        'Field',
        'Alias',
        'Cast',
        'IfElse',
        'SimpleCase',
        'SearchedCase',
        'Coalesce',
        'NullIf',
        'Min',
        'Max',
        'Any',
        'All',
        'Lag',
        'Lead',
        'FirstValue',
        'LastValue',
        'First',
        'Last',
        'WindowFunction',
        'ScalarSubquery',
    ]
)


class TeradataCompiler(SQLGlotCompiler):
    __slots__ = ()
    dialect = TeradataDialect
    type_mapper = TeradataType
    supports_qualify = True
    rewrites = (
        exclude_unsupported_window_frame_from_rank,
        exclude_unsupported_window_frame_from_row_number,
        *SQLGlotCompiler.rewrites,
    )
    UNSUPPORTED_OPS = tuple(
        op
        for op in ALL_OPERATIONS
        if issubclass(op, ops.Value) and op.__name__ not in _SUPPORTED_VALUES
    )
    LOWERED_OPS: ClassVar[dict] = {
        ops.Sample: None,
        ops.Bucket: None,
        ops.Capitalize: None,
    }
    SIMPLE_OPS: ClassVar[dict] = {
        ops.StringLength: 'char_length',
        ops.StringReplace: 'oreplace',
    }

    @staticmethod
    def _predicate(value):
        return value.meta.get('ibis_teradata_predicate', value.eq(1)).copy()

    @staticmethod
    def _boolean_value(predicate):
        # CASE WHEN ... ELSE 0 would lose SQL's third truth value.
        result = sge.Case(
            ifs=[
                sge.If(this=predicate.copy(), true=sge.convert(1)),
                sge.If(this=sg.not_(predicate.copy()), true=sge.convert(0)),
            ],
            default=NULL,
        )
        result.meta['ibis_teradata_predicate'] = predicate.copy()
        return result

    def visit_node(self, op, **kwargs):
        name = type(op).__name__
        if name in ('And', 'Or', 'Xor'):
            kwargs = {key: self._predicate(value) for key, value in kwargs.items()}
        elif name == 'Not':
            kwargs['arg'] = self._predicate(kwargs['arg'])
        elif name == 'IfElse':
            kwargs['bool_expr'] = self._predicate(kwargs['bool_expr'])
        elif name == 'SearchedCase':
            kwargs['cases'] = tuple(map(self._predicate, kwargs['cases']))
        for key in ('predicates', 'qualified'):
            if key in kwargs:
                kwargs[key] = tuple(map(self._predicate, kwargs[key]))
        if isinstance(op, ops.Reduction) and kwargs.get('where') is not None:
            kwargs['where'] = self._predicate(kwargs['where'])
        result = super().visit_node(op, **kwargs)
        if (
            isinstance(op, ops.Value)
            and op.dtype.is_boolean()
            and name not in _STORED_BOOLEANS
        ):
            return self._boolean_value(result)
        return result

    def visit_NonNullLiteral(self, op, *, value, dtype):
        if dtype.is_boolean():
            return sge.convert(int(value))
        if dtype.is_interval():
            return self._integer_to_interval(sge.convert(value), dtype.unit)
        if dtype.is_floating() and not math.isfinite(value):
            raise exc.UnsupportedOperationError(
                'Teradata does not support NaN or infinity literals'
            )
        if dtype.is_binary():
            raise exc.UnsupportedOperationError('Upload binary values using a memtable')
        if dtype.is_timestamp():
            TeradataType.to_string(dtype)
        return super().visit_NonNullLiteral(op, value=value, dtype=dtype)

    def visit_Cast(self, op, *, arg, to):
        if to.is_boolean():
            return (
                arg
                if op.arg.dtype.is_boolean()
                else self._boolean_value(sge.NEQ(this=arg, expression=sge.convert(0)))
            )
        if op.arg.dtype.is_boolean():
            return self.cast(arg, to)
        return super().visit_Cast(op, arg=arg, to=to)

    def visit_Xor(self, op, *, left, right):
        return sg.and_(sg.or_(left, right), sg.not_(sg.and_(left, right)))

    def visit_IdenticalTo(self, op, *, left, right):
        return sg.or_(
            sg.and_(sg.not_(left.is_(NULL)), sg.not_(right.is_(NULL)), left.eq(right)),
            sg.and_(left.is_(NULL), right.is_(NULL)),
        )

    def visit_Any(self, op, *, arg, where):
        return self.agg.max(arg, where=where)

    def visit_All(self, op, *, arg, where):
        return self.agg.min(arg, where=where)

    def visit_Min(self, op, *, arg, where):
        return self.agg.min(arg, where=where)

    def visit_Max(self, op, *, arg, where):
        return self.agg.max(arg, where=where)

    def visit_Sum(self, op, *, arg, where):
        if op.arg.dtype.is_integer() or op.arg.dtype.is_boolean():
            arg = self.cast(arg, op.dtype)
        return self.agg.sum(arg, where=where)

    def visit_Mean(self, op, *, arg, where):
        if op.dtype.is_floating():
            arg = self.cast(arg, dt.float64)
        return self.agg.avg(arg, where=where)

    def visit_Count(self, op, *, arg, where):
        return self.cast(self.agg.count(arg, where=where), dt.int64)

    def visit_CountStar(self, op, *, arg, where):
        return self.cast(self.agg.count(STAR, where=where), dt.int64)

    def visit_CountDistinct(self, op, *, arg, where):
        return self.cast(
            self.agg.count(sge.Distinct(expressions=[arg]), where=where), dt.int64
        )

    def visit_Divide(self, op, *, left, right):
        if op.dtype.is_floating():
            left = self.cast(left, dt.float64)
        return sge.paren(left) / sge.paren(right)

    def visit_FloorDivide(self, op, *, left, right):
        return self.cast(
            self.f.floor(sge.paren(self.cast(left, dt.float64)) / sge.paren(right)),
            op.dtype,
        )

    def visit_Date(self, op, *, arg):
        return self.cast(arg, dt.date)

    def _integer_to_interval(self, arg, unit):
        unit = unit.short
        if unit in ('Y', 'Q', 'M'):
            months = (
                sge.paren(self.cast(arg, dt.int64)) * {'Y': 12, 'Q': 3}[unit]
                if unit != 'M'
                else arg.copy()
            )
            result = self.f.anon.numtoyminterval(months, 'MONTH')
            result.meta['ibis_teradata_months'] = months.copy()
            return result
        if unit in ('W', 'D'):
            days = sge.paren(self.cast(arg, dt.int64)) * 7 if unit == 'W' else arg
            # Explicit leading precision allows lookbacks above 99 days.
            return sge.Cast(
                this=days,
                to=sge.DataType(
                    this=sge.DataType.Type.INTERVAL,
                    expressions=[
                        sge.Anonymous(this='DAY', expressions=[sge.convert(4)])
                    ],
                ),
            )
        if unit in ('ms', 'us'):
            factor = Decimal('0.001') if unit == 'ms' else Decimal('0.000001')
            seconds = sge.paren(self.cast(arg, dt.Decimal(20, 0))) * sge.convert(factor)
            return self.f.anon.numtodsinterval(seconds, 'SECOND')
        if unit in ('h', 'm', 's'):
            return self.f.anon.numtodsinterval(
                arg, {'h': 'HOUR', 'm': 'MINUTE', 's': 'SECOND'}[unit]
            )
        raise exc.UnsupportedOperationError(
            'Teradata intervals support microsecond precision; '
            f'interval unit {unit!r} is unsupported'
        )

    def visit_IntervalFromInteger(self, op, *, arg, unit):
        return self._integer_to_interval(arg, unit)

    def _calendar_arithmetic(self, op, left, right, sign):
        unit = op.right.dtype.unit.short
        if unit in ('M', 'Q', 'Y'):
            if isinstance(op.right, ops.Literal):
                months = sge.convert(
                    sign * op.right.value * {'M': 1, 'Q': 3, 'Y': 12}[unit]
                )
            elif (count := right.meta.get('ibis_teradata_months')) is not None:
                # ADD_MONTHS requires an INTEGER count, rather than BIGINT.
                months = self.cast(count.copy(), dt.int32)
                if sign < 0:
                    months = sge.Neg(this=months)
            else:
                raise exc.UnsupportedOperationError(
                    'Calendar arithmetic requires a literal interval or '
                    'an integer expression converted with as_interval()'
                )
            return self.f.anon.add_months(left, months)
        return self.binop(sge.Add if sign == 1 else sge.Sub, left, right)

    def visit_DateAdd(self, op, *, left, right):
        return self._calendar_arithmetic(op, left, right, 1)

    visit_TimestampAdd = visit_DateAdd

    def visit_DateSub(self, op, *, left, right):
        return self._calendar_arithmetic(op, left, right, -1)

    visit_TimestampSub = visit_DateSub

    def visit_Log(self, op, *, arg, base):
        return self.f.ln(arg) if base is None else self.f.ln(arg) / self.f.ln(base)

    def visit_Log2(self, op, *, arg):
        return self.f.ln(arg) / self.f.ln(2)

    def visit_StringContains(self, op, *, haystack, needle):
        return self.f.strpos(haystack, needle) > 0

    def visit_StringConcat(self, op, *, arg):
        return reduce(lambda left, right: sge.DPipe(this=left, expression=right), arg)

    def visit_StringJoin(self, op, *, sep, arg):
        # CONCAT_WS semantics: skip null arguments, retain empty strings, and
        # return NULL for a null separator. Teradata concatenates using ||.
        result = arg[0]
        for value in arg[1:]:
            combined = self.visit_StringConcat(
                op, arg=(result.copy(), sep.copy(), value.copy())
            )
            result = self.if_(
                result.is_(NULL), value, self.if_(value.is_(NULL), result, combined)
            )
        return self.if_(sep.is_(NULL), NULL, self.f.coalesce(result, ''))

    def visit_StartsWith(self, op, *, arg, start):
        return self.f.substr(arg, 1, self.f.char_length(start)).eq(start)

    def visit_EndsWith(self, op, *, arg, end):
        return self.f.right(arg, self.f.char_length(end)).eq(end)

    def visit_StringSQLILike(self, op, *, arg, pattern, escape):
        return self.visit_StringSQLLike(
            op, arg=self.f.lower(arg), pattern=self.f.lower(pattern), escape=escape
        )

    def visit_Limit(self, op, *, parent, n, offset):
        if not isinstance(offset, int) or (n is not None and not isinstance(n, int)):
            raise exc.UnsupportedOperationError(
                'Teradata limits require integer bounds'
            )
        result = super().visit_Limit(op, parent=parent, n=None, offset=0)
        inner = result.this if isinstance(result, sge.Subquery) else result
        if n is None and not offset:
            return result
        if inner.args.get('distinct'):
            raise exc.UnsupportedOperationError(
                'Materialize a distinct result before applying a limit'
            )
        order = inner.args.get('order')
        if order is None:
            first = inner.selects[0]
            if isinstance(first, sge.Alias):
                first = first.this
            if isinstance(first, sge.Star):
                first = sg.column(op.parent.schema.names[0], quoted=True)
            order = sge.Order(expressions=[sge.Ordered(this=first.copy())])
        row = sge.Window(this=sge.RowNumber(), order=order.copy())
        predicates = [row.copy() > offset] if offset else []
        if n is not None:
            predicates.append(row <= offset + n)
        inner.qualify(*predicates, copy=False)
        return result


compiler = TeradataCompiler()
