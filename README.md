# ibis-framework-teradata

Use ibis to join local data with tables on Teradata. This is not feature complete.

- `ibis.memtable()` -> temporary (volatile) Teradata tables, uploaded automatically
- Polars, pandas or Arrow -> local data inputs
- Local + remote joins -> run on Teradata, using the same connection as the upload
- Close the connection -> temporary tables dropped

## Install

```powershell
pip install 'ibis-framework-teradata'
```

- Python 3.11+; Ibis 12
- After installing, use `ibis.teradata` (no separate package import required)

## Join local data to Teradata

```python
import os
import ibis
import polars as pl

lookup = pl.DataFrame({
    'customer_id': [101, 205],
    'segment': ['retail', 'partner'],
})

with ibis.teradata.connect(
        host=os.environ['TD_HOST'],
        user=os.environ['TD_USER'],
        password=os.environ['TD_PASSWORD'],
        database='analytics',
) as con:
    sales = con.table('sales')
    local = ibis.memtable(lookup)
    joined = sales.join(local, 'customer_id')
    query = joined.group_by('segment').aggregate(total=joined.amount.sum())
    result = con.to_polars(query)
```

## Handy examples

Inside an open connection (i.e., in the above `with` block):

```python
sql = con.compile(query)  # Preview the SQL
frame = con.execute(query)  # pandas result
arrow = con.to_pyarrow(query)  # Arrow result
con.cleanup_memtables()  # Clear uploads if keeping the connection open
```

A date lookback:

```python
events = con.table('events')
lookback = ibis.literal(7).as_interval('D')
query = events.filter(
    events.dt.between(events.index_date - lookback, events.index_date)
)
result = con.to_polars(query)
```

Trailing spaces in remote text:

```python
customers = con.table('customers')
customers = customers.mutate(code=customers.code.rstrip())
```

## Good to know

- Includes common operations: joins, filters, grouping, strings and date calculations
- Most advanced ibis features aren't supported (yet?)
- Day lookbacks: up to 9999 days in either direction
- Time precision: microseconds; nanoseconds aren't supported
