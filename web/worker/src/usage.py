from __future__ import annotations

from typing import Any


RESERVE_CALL_SQL = """
INSERT INTO monthly_usage (month_utc, calls)
VALUES (?, 1)
ON CONFLICT(month_utc) DO UPDATE SET calls = calls + 1
WHERE monthly_usage.calls < ?
RETURNING calls
"""


async def reserve_monthly_call(
    database: Any,
    month_utc: str,
    monthly_limit: int,
) -> bool:
    """Atomically reserve one provider call; count attempts, including failures."""
    result = await (
        database.prepare(RESERVE_CALL_SQL)
        .bind(month_utc, monthly_limit)
        .first()
    )
    return result is not None
