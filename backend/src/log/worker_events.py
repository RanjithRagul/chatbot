from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def log_event(event: str, **fields: Any) -> None:
    """Write metadata-only structured events to the Worker runtime log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    print(json.dumps(record, separators=(",", ":"), default=str))
