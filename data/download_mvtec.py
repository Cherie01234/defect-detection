"""Fetch / unpack one MVTec AD category into data/mvtec/<category>.

MVTec AD requires accepting their (research-only) license, so this script does
not bundle a hard-coded download link. Use one of:

  # A) you already downloaded e.g. bottle.tar.xz from the MVTec AD page
  python data/download_mvtec.py --archive path/to/bottle.tar.xz

  # B) you have a direct URL to the category archive
  python data/download_mvtec.py --url https://.../bottle.tar.xz --category bottle

Get the dataset here (per-object archives are offered on the page):
  https://www.mvtec.com/company/research/datasets/mvtec-ad

After this, the category sits in MVTec's standard layout and works directly:
  python eval/run_eval.py --root data/mvtec/bottle --backbone wide_resnet50_2
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MVTEC_DIR = ROOT / "data" / "mvtec"


def _extract(archive: Path) -> None:
    MVTEC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} -> {MVTEC_DIR} ...")
    with tarfile.open(archive) as tar:
        tar.extractall(MVTEC_DIR)  # archives unpack to <category>/...
    print("Done. Categories now under data/mvtec/:")
    for p in sorted(MVTEC_DIR.iterdir()):
        if p.is_dir():
            print(f"  - {p.name}")


def _download(url: str, category: str) -> Path:
    MVTEC_DIR.mkdir(parents=True, exist_ok=True)
    dest = MVTEC_DIR / f"{category}.tar.xz"
    print(f"Downloading {url}\n  -> {dest}")

    def hook(block, size, total):
        if total > 0:
            pct = min(100, block * size * 100 // total)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", help="path to an already-downloaded *.tar.xz")
    parser.add_argument("--url", help="direct URL to a category *.tar.xz")
    parser.add_argument("--category", default="bottle")
    args = parser.parse_args()

    if args.archive:
        _extract(Path(args.archive))
    elif args.url:
        _extract(_download(args.url, args.category))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
