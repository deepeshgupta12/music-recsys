from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Iterable, List, Optional


from musicrec.storage.feature_table import load_feature_table
from musicrec.storage.tag_store import TagStore, TagStoreConfig
from musicrec.tagging.tagger import Tagger, TaggerConfig


def _is_dictlike(x: Any) -> bool:
    return hasattr(x, "items") and callable(getattr(x, "items"))


def _maybe_pandas_records(obj: Any) -> Optional[List[Dict[str, Any]]]:
    # pandas.DataFrame -> records
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    try:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")  # type: ignore[arg-type]
    except Exception:
        return None
    return None


def _maybe_pyarrow_records(obj: Any) -> Optional[List[Dict[str, Any]]]:
    # pyarrow.Table -> list of dicts
    try:
        import pyarrow as pa  # type: ignore
    except Exception:
        return None

    try:
        if isinstance(obj, pa.Table):
            return obj.to_pylist()
    except Exception:
        return None
    return None


def _maybe_duckdb_records(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """
    DuckDBPyRelation common methods:
      - df() / fetchdf()
      - fetchall() + description / columns
    We avoid importing duckdb explicitly; rely on duckdb relation methods.
    """
    # Relation -> pandas df if available
    for m in ("df", "fetchdf"):
        fn = getattr(obj, m, None)
        if callable(fn):
            try:
                df = fn()
                recs = _maybe_pandas_records(df)
                if recs is not None:
                    return recs
            except Exception:
                pass

    # Relation -> fetchall() + columns/description
    fetchall = getattr(obj, "fetchall", None)
    if callable(fetchall):
        try:
            rows = fetchall()
            cols = None
            # duckdb relation: .columns sometimes exists
            cols_attr = getattr(obj, "columns", None)
            if isinstance(cols_attr, list) and cols_attr and all(isinstance(c, str) for c in cols_attr):
                cols = cols_attr
            # fallback: description like DB-API
            desc = getattr(obj, "description", None)
            if cols is None and desc:
                try:
                    cols = [d[0] for d in desc]
                except Exception:
                    cols = None

            if cols and rows:
                out: List[Dict[str, Any]] = []
                for r in rows:
                    out.append({cols[i]: r[i] for i in range(min(len(cols), len(r)))})
                return out
        except Exception:
            pass

    return None


def _coerce_feature_table_to_rows(ft: Any) -> List[Dict[str, Any]]:
    """
    Convert whatever load_feature_table() returns into a list[dict] rows.
    Supports:
      - pandas DataFrame
      - pyarrow Table
      - duckdb relation
      - list/tuple of dicts
      - dict-of-dicts keyed by track_id
      - objects with .rows / .data / .to_dict(orient="records")
      - generic iterables yielding dicts
    """
    if ft is None:
        raise RuntimeError("load_feature_table() returned None")

    # 1) common explicit attrs
    for attr in ("rows", "data"):
        v = getattr(ft, attr, None)
        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
            return v

    # 2) pandas
    recs = _maybe_pandas_records(ft)
    if recs is not None:
        return recs

    # 3) pyarrow
    recs = _maybe_pyarrow_records(ft)
    if recs is not None:
        return recs

    # 4) duckdb relation (or similar)
    recs = _maybe_duckdb_records(ft)
    if recs is not None:
        return recs

    # 5) dict-of-dicts keyed by track_id
    if _is_dictlike(ft) and not isinstance(ft, list):
        try:
            out = []
            for k, v in ft.items():
                if isinstance(v, dict):
                    row = {"track_id": k}
                    row.update(v)
                    out.append(row)
            if out:
                return out
        except Exception:
            pass

    # 6) list/tuple already
    if isinstance(ft, (list, tuple)):
        if not ft:
            return []
        if isinstance(ft[0], dict):
            return list(ft)  # type: ignore[return-value]

    # 7) object exposes to_dict(orient="records")
    to_dict = getattr(ft, "to_dict", None)
    if callable(to_dict):
        try:
            recs = to_dict(orient="records")
            if isinstance(recs, list) and (not recs or isinstance(recs[0], dict)):
                return recs
        except Exception:
            pass

    # 8) generic iterable yielding dicts
    if hasattr(ft, "__iter__"):
        try:
            out = []
            for i, r in enumerate(ft):
                if isinstance(r, dict):
                    out.append(r)
                else:
                    # stop early if it’s not dict-like
                    break
                if i >= 5 and out:
                    # enough proof it’s dict rows
                    return out + [x for x in ft if isinstance(x, dict)]  # type: ignore[misc]
            if out:
                return out
        except Exception:
            pass

    # If nothing worked, raise with actionable debug
    t = type(ft).__name__
    mod = getattr(type(ft), "__module__", "")
    sample_attrs = [a for a in ("rows", "data", "df", "fetchdf", "fetchall", "to_dict", "columns") if hasattr(ft, a)]
    raise RuntimeError(
        "Unsupported feature_table structure from load_feature_table(). "
        f"type={mod}.{t} supported_attrs_found={sample_attrs}. "
        "Update run_tagging.py coercion for this structure."
    )


def _split_features_vs_meta(row: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Keep this tolerant: feature keys are best-effort; everything else is meta.
    """
    feature_keys = {
        "energy",
        "danceability",
        "tempo",
        "instrumentalness",
        "loudness",
        "duration_ms",
        "key",
        "mode",
    }

    features: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    for k, v in row.items():
        if k == "track_id":
            continue
        if k in feature_keys:
            features[k] = v
        else:
            meta[k] = v

    return features, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="heuristic", choices=["heuristic", "openai"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all tracks")
    ap.add_argument("--db-path", default="runtime/tags.db")
    args = ap.parse_args()

    ft = load_feature_table()
    rows = _coerce_feature_table_to_rows(ft)

    store = TagStore(TagStoreConfig(db_path=args.db_path))
    tagger = Tagger(TaggerConfig(provider=args.provider))

    n_scanned = 0
    n_written = 0
    t0 = time.time()

    for row in rows:
        n_scanned += 1
        if args.limit and n_written >= args.limit:
            break

        track_id = str(row.get("track_id") or "").strip()
        if not track_id:
            continue

        features, meta = _split_features_vs_meta(row)
        bundle = tagger.tag_track(track_id=track_id, features=features, meta=meta)
        store.upsert(bundle)
        n_written += 1

    dt = time.time() - t0
    st = store.stats()
    print(
        f"tagging_done provider={args.provider} wrote={n_written} scanned={n_scanned} "
        f"secs={dt:.2f} store_rows={st['rows']} last_updated_at={st['last_updated_at']}"
    )


if __name__ == "__main__":
    main()