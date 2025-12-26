from __future__ import annotations

import json

from musicrec.config import get_paths
from musicrec.ingest import ingest_catalog


def main() -> int:
    paths = get_paths()
    in_csv = paths.raw_catalog_csv
    out_parquet = paths.data_processed_dir / "catalog.parquet"
    report_json = paths.data_processed_dir / "catalog_ingest_report.json"

    if not in_csv.exists():
        print("ERROR: input CSV not found:", str(in_csv))
        return 2

    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)

    df, report = ingest_catalog(str(in_csv))

    if report.missing_required_columns:
        print("ERROR: missing required columns:", report.missing_required_columns)
        return 3

    # Hard fail if too many invalid dates (sanity guard)
    if report.invalid_release_date_rows > 0:
        print(f"WARNING: invalid release_date rows: {report.invalid_release_date_rows}")

    df.to_parquet(out_parquet, index=False)

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)

    print("OK: wrote parquet:", str(out_parquet))
    print("OK: wrote report:", str(report_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())