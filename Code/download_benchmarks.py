"""Download CrowS-Pairs and StereoSet, convert StereoSet to a flat CSV.
Run from project root: python Code/download_benchmarks.py
"""
import json
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Dataset"
OUT.mkdir(parents=True, exist_ok=True)

URLS = {
    "CrowS-Pairs.csv": [
        "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
        "https://raw.githubusercontent.com/nyu-mll/crows-pairs/main/data/crows_pairs_anonymized.csv",
    ],
    "StereoSet.json": [
        "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json",
        "https://raw.githubusercontent.com/moinnadeem/StereoSet/main/data/dev.json",
    ],
}

def download(name, urls):
    dest = OUT / name
    for url in urls:
        try:
            print(f"Downloading {name} from {url}", flush=True)
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as response:
                content = response.read()
            if len(content) < 100:
                raise ValueError("File is unexpectedly small")
            dest.write_bytes(content)
            print(f"Saved: {dest} ({len(content):,} bytes)")
            return dest
        except Exception as exc:
            print(f"  Failed: {exc}")
    raise RuntimeError(f"Could not download {name}. Check your internet connection or source URL.")

def convert_stereoset(path):
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    rows = []
    for task in ("intrasentence", "intersentence"):
        for item in raw["data"].get(task, []):
            options = {s["gold_label"]: s["sentence"] for s in item.get("sentences", [])}
            if not all(k in options for k in ("stereotype", "anti-stereotype")):
                continue
            rows.append({
                "task": task,
                "context": item.get("context", ""),
                "bias_type": item.get("bias_type", ""),
                "target": item.get("target", ""),
                "stereotype": options["stereotype"],
                "anti_stereotype": options["anti-stereotype"],
                "unrelated": options.get("unrelated", ""),
            })
    if not rows:
        raise ValueError("No StereoSet sentence pairs found")
    dest = OUT / "StereoSet.csv"
    pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"Saved: {dest} ({len(rows):,} rows)")

def main():
    errors = []
    for name, urls in URLS.items():
        try:
            path = download(name, urls)
            if name == "CrowS-Pairs.csv":
                df = pd.read_csv(path)
                missing = {"sent_more", "sent_less"} - set(df.columns)
                if missing:
                    raise ValueError(f"CrowS-Pairs missing columns: {missing}")
                print(f"CrowS-Pairs: {len(df):,} rows")
            else:
                convert_stereoset(path)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"ERROR: {exc}")
    if errors:
        raise SystemExit("Some downloads failed:\n" + "\n".join(errors))
    print("DONE. Files ready in Dataset/")

if __name__ == "__main__":
    main()
