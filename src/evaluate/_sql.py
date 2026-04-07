"""SQL normalization, code-fence stripping, and syntax validation helpers."""

from __future__ import annotations

import re
import sqlite3

_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    """Collapse whitespace and lowercase *sql* for comparison."""
    return re.sub(r"\s+", " ", sql.strip().lower())


def strip_code_fence(text: str) -> str:
    """Extract raw SQL from a markdown code fence and strip trailing explanation text."""
    m = _CODE_FENCE_RE.search(text)
    sql = m.group(1).strip() if m else text.strip()
    # Drop continuation markers the model sometimes appends after the SQL.
    for sentinel in ("\n\n", "\n###", "\n--"):
        idx = sql.find(sentinel)
        if idx != -1:
            sql = sql[:idx].strip()
    return sql


def is_valid_sql(ddl: str, sql: str) -> bool:
    """Return True if *sql* can be EXPLAINed against a schema built from *ddl*."""
    try:
        con = sqlite3.connect(":memory:")
        con.executescript(ddl)
        con.execute(f"EXPLAIN {sql}")
        con.close()
        return True
    except sqlite3.Error:
        return False
