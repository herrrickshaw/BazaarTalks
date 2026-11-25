#!/usr/bin/env python3
"""
Dated, checksummed inventory of the market-data tree — the "before" record for
integrity auditing after size-reduction (git gc / lfs prune / DuckDB migration).

Captures, per file: path, bytes, mtime (UTC), sha256, and row/col counts for
parquet/csv/duckdb. LFS-tracked files reuse the LFS OID (already a sha256) so we
don't re-hash what git has already content-addressed.

Writes  <outdir>/data_manifest_<YYYYMMDD>.json  (+ a .csv summary)
Re-runnable: `--verify <old.json>` diffs a later run against an earlier manifest.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path("/Users/umashankar")

# Trees that hold real market data. bhavcopy_cache is the crown jewel (raw NSE/BSE
# EOD — expensive to refetch); the scan dirs are regenerable but we still baseline
# them so a later audit can prove exactly what changed.
TARGETS = [
    HOME / "Downloads" / "data",
    HOME / "data",
]

# "audit" holds this script's own output — scanning it makes every verify report
# the manifest itself as changed, which is noise that hides real drift.
SKIP_DIRS = {"__pycache__", ".git", "ohlcv.lmdb", "audit"}
HASH_MAX_BYTES = 2 * 1024**3  # hash files up to 2 GB; larger get size+mtime only


def sha256_of(p: Path) -> str | None:
    if p.stat().st_size > HASH_MAX_BYTES:
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lfs_oids() -> dict[str, str]:
    """path -> sha256 oid, straight from git-lfs (free checksums)."""
    out: dict[str, str] = {}
    try:
        r = subprocess.run(
            ["git", "lfs", "ls-files", "-l"],
            cwd=HOME, capture_output=True, text=True, timeout=180,
        )
        for line in r.stdout.splitlines():
            # "<oid> * <path>"  or  "<oid> - <path>"
            parts = line.split(None, 2)
            if len(parts) == 3:
                out[parts[2].strip()] = parts[0].strip()
    except Exception as e:
        print(f"  (lfs ls-files unavailable: {e})", file=sys.stderr)
    return out


def shape_of(p: Path) -> dict:
    """Row/col counts for tabular files — the thing that actually proves nothing was lost."""
    s = p.suffix.lower()
    try:
        if s == ".parquet":
            import pyarrow.parquet as pq
            md = pq.ParquetFile(p).metadata
            return {"rows": md.num_rows, "cols": md.num_columns}
        if s == ".csv":
            with p.open("r", errors="replace") as fh:
                first = fh.readline()
                n = sum(1 for _ in fh)
            return {"rows": n, "cols": first.count(",") + 1 if first else 0}
        if s == ".duckdb":
            import duckdb
            con = duckdb.connect(str(p), read_only=True)
            tabs = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            shp = {}
            for t in tabs:
                n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
                c = len(con.execute(f'DESCRIBE "{t}"').fetchall())
                shp[t] = {"rows": n, "cols": c}
            con.close()
            return {"tables": shp}
    except Exception as e:
        return {"shape_error": str(e)[:120]}
    return {}


def walk(root: Path, oids: dict[str, str]) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            rel = str(p.relative_to(HOME))
            rec = {
                "path": rel,
                "bytes": st.st_size,
                "mtime_utc": dt.datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
            }
            if rel in oids:
                rec["sha256"] = oids[rel]
                rec["sha256_src"] = "lfs-oid"
            else:
                try:
                    d = sha256_of(p)
                    rec["sha256"] = d
                    rec["sha256_src"] = "computed" if d else "skipped-too-large"
                except OSError as e:
                    rec["sha256"] = None
                    rec["sha256_src"] = f"error:{e}"
            if p.suffix.lower() in {".parquet", ".csv", ".duckdb"}:
                rec.update(shape_of(p))
            rows.append(rec)
    return rows


def lfs_object_inventory() -> dict:
    """Which LFS objects exist in the LOCAL cache, and are they on the remote?

    This is the part `git lfs prune` actually deletes. An object that is local-only
    (never pushed) is UNRECOVERABLE once pruned — that's the risk this records.
    """
    inv = {"local_objects": 0, "local_bytes": 0, "referenced_oids": 0}
    objdir = HOME / ".git" / "lfs" / "objects"
    if objdir.exists():
        n = b = 0
        for dp, _, fns in os.walk(objdir):
            for fn in fns:
                try:
                    b += (Path(dp) / fn).stat().st_size
                    n += 1
                except OSError:
                    pass
        inv["local_objects"], inv["local_bytes"] = n, b
    inv["referenced_oids"] = len(lfs_oids())
    return inv


def build(outdir: Path) -> Path:
    stamp = dt.date.today().strftime("%Y%m%d")
    oids = lfs_oids()
    print(f"  lfs oids available: {len(oids)}")
    files: list[dict] = []
    for t in TARGETS:
        print(f"  scanning {t} …")
        got = walk(t, oids)
        print(f"    {len(got)} files")
        files.extend(got)
    files.sort(key=lambda r: r["path"])

    manifest = {
        "generated_utc": dt.datetime.utcnow().isoformat() + "Z",
        "host": os.uname().nodename,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HOME,
            capture_output=True, text=True).stdout.strip(),
        "purpose": "pre-size-reduction integrity baseline (git gc / lfs prune / duckdb migration)",
        "targets": [str(t) for t in TARGETS],
        "lfs": lfs_object_inventory(),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / f"data_manifest_{stamp}.json"
    jp.write_text(json.dumps(manifest, indent=1))

    cp = outdir / f"data_manifest_{stamp}.csv"
    with cp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "mtime_utc", "sha256", "sha256_src", "rows", "cols"])
        for f in files:
            w.writerow([f["path"], f["bytes"], f["mtime_utc"], f.get("sha256", ""),
                        f.get("sha256_src", ""), f.get("rows", ""), f.get("cols", "")])
    return jp


def verify(old_path: Path, outdir: Path) -> int:
    old = json.loads(old_path.read_text())
    oldmap = {f["path"]: f for f in old["files"]}
    oids = lfs_oids()
    cur: list[dict] = []
    for t in TARGETS:
        cur.extend(walk(t, oids))
    curmap = {f["path"]: f for f in cur}

    missing = sorted(set(oldmap) - set(curmap))
    added = sorted(set(curmap) - set(oldmap))
    changed = [p for p in set(oldmap) & set(curmap)
               if oldmap[p].get("sha256") and curmap[p].get("sha256")
               and oldmap[p]["sha256"] != curmap[p]["sha256"]]

    print(f"\n=== VERIFY vs {old_path.name} (baseline {old['generated_utc']}) ===")
    print(f"  baseline files : {len(oldmap)}")
    print(f"  current  files : {len(curmap)}")
    print(f"  MISSING        : {len(missing)}")
    for p in missing[:15]:
        print(f"      - {p}")
    print(f"  changed sha256 : {len(changed)}")
    for p in changed[:10]:
        print(f"      ~ {p}")
    print(f"  added          : {len(added)}")
    return 1 if missing else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HOME / "data" / "audit"))
    ap.add_argument("--verify", metavar="OLD_JSON")
    a = ap.parse_args()
    od = Path(a.outdir)
    if a.verify:
        sys.exit(verify(Path(a.verify), od))
    p = build(od)
    m = json.loads(p.read_text())
    print(f"\n  manifest → {p}")
    print(f"  {m['n_files']} files, {m['total_bytes']/1024**3:.2f} GB")
