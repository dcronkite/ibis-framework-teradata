from __future__ import annotations

import ibis
import pytest

from ibis_framework_teradata import Backend


def memtable(data, *, name=None, **kwargs):
    table = ibis.memtable(data, **kwargs)
    return table.op().copy(name=name).to_expr() if name is not None else table


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False
        self.rows = []
        self.description = None
        self.columntypename = None

    def execute(self, query, params=None, **kwargs):
        self.connection.statements.append((query, params, kwargs))
        if self.connection.fail and self.connection.fail(query):
            raise RuntimeError('simulated execution failure')
        for match, response in self.connection.responses.items():
            if match in query:
                self.rows, self.description, self.columntypename = response
                self.rows = list(self.rows)
                break
        return self

    def executemany(self, query, rows):
        self.connection.batches.append((query, rows))
        if self.connection.fail_batch:
            raise RuntimeError('simulated upload failure')

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchmany(self, size):
        rows, self.rows = self.rows[:size], self.rows[size:]
        return rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.autocommit = True
        self.closed = False
        self.statements = []
        self.batches = []
        self.cursors = []
        self.responses = {}
        self.fail = None
        self.fail_batch = False

    def cursor(self):
        cursor = FakeCursor(self)
        self.cursors.append(cursor)
        return cursor

    def close(self):
        self.closed = True


@pytest.fixture
def connection():
    return FakeConnection()


@pytest.fixture
def backend(connection):
    return Backend.from_connection(connection, batch_size=2)
