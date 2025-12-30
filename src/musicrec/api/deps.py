from __future__ import annotations

import os
from functools import lru_cache

from musicrec.storage.feedback_store import FeedbackStore


@lru_cache(maxsize=1)
def get_feedback_store() -> FeedbackStore:
    # Keep DB under .cache but never commit it (we already ignored .cache/)
    db_path = os.getenv("MUSICREC_FEEDBACK_DB", ".cache/feedback/feedback.sqlite3")
    return FeedbackStore(db_path=db_path)