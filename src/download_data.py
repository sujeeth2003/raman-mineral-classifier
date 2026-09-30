"""Download the public RRUFF Raman archives (about 420 MB) and unzip them into data/raw/unz."""
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://rruff.info/zipped_data_files/raman/"
ARCHIVES = ["excellent_unoriented", "excellent_oriented", "fair_unoriented", "unrated_unoriented", "unrated_oriented"]

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        zpath = RAW / f"{name}.zip"
        if not zpath.exists():
            print("downloading", name)
            urllib.request.urlretrieve(BASE + name + ".zip", zpath)
        zipfile.ZipFile(zpath).extractall(RAW / "unz")
    print("done")
