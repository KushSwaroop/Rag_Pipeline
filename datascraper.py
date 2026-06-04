"""
datascraper.py - Downloads 30 documents (PDFs + CSVs) from Kaggle and GovData.de
into the ./PDF directory for ingestion into the RAG pipeline.

Sources:
  - Kaggle API  (~15 datasets) - requires KAGGLE_API_TOKEN env var or ~/.kaggle/access_token
  - GovData.de  (~15 datasets) - public CKAN API, no auth required

Usage:
  python datascraper.py                   # download all 30
  python datascraper.py --kaggle-only     # Kaggle only
  python datascraper.py --govdata-only    # GovData.de only
"""

import os
import sys
import json
import time
import zipfile
import argparse
import requests
from pathlib import Path

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
DOWNLOAD_DIR = Path(__file__).parent / "DATA"
MAX_FILE_SIZE_MB = 100  # skip files larger than this
REQUEST_TIMEOUT = 60    # seconds per download

# Diverse Kaggle datasets across mixed topics (owner/dataset-name)
KAGGLE_DATASETS = [
    # AI / Machine Learning
    "kaggle/meta-learner-dataset",
    "shivamb/netflix-shows",
    # Finance
    "dgawlik/nyse",
    "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
    # Health
    "rashikrahmanpritom/heart-attack-analysis-prediction-dataset",
    "johnsmith88/heart-disease-dataset",
    "uciml/pima-indians-diabetes-database",
    # Climate / Environment
    "berkeleyearth/climate-change-earth-surface-temperature-data",
    "sevgisarac/temperature-change",
    # Education
    "spscientist/students-performance-in-exams",
    "aljarah/xAPI-Edu-Data",
    # General / Social
    "unsdsn/world-happiness",
    "mrmorj/dataset-of-songs-in-spotify",
    "datasnaek/youtube-new",
    "carrie1/ecommerce-data",
]

# GovData.de search queries covering diverse government topics
GOVDATA_SEARCH_QUERIES = [
    "bildung schule",          # education
    "gesundheit krankenhaus",  # health
    "umwelt klima",            # climate / environment
    "verkehr mobilitat",       # transport
    "finanzen haushalt",       # finance / budget
    "bevolkerung demografie",  # demographics
    "energie erneuerbar",      # energy
    "wirtschaft arbeitsmarkt", # economy / jobs
    "kultur museum",           # culture
    "wohnen miete",            # housing
    "landwirtschaft",          # agriculture
    "tourismus",               # tourism
    "kriminalitat sicherheit", # crime / safety
    "digitalisierung",         # digitalization
    "soziales",                # social affairs
]

GOVDATA_API_BASE = "https://ckan.govdata.de/api/3/action"
ALLOWED_FORMATS = {"pdf", "csv", "xlsx", "xls"}


# -----------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------
def ensure_download_dir():
    """Create the download directory if it doesn't exist."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Remove problematic characters from filenames."""
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        name = name.replace(ch, '_')
    return name[:200]  # cap length


def file_already_exists(filename: str) -> bool:
    """Check if a file (by name) already exists in the download dir."""
    return (DOWNLOAD_DIR / filename).exists()


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """Stream-download a file from a URL. Returns True on success."""
    try:
        print(f"  [DOWN] Downloading: {description or url}")
        resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()

        # Check content length if available
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_FILE_SIZE_MB * 1024 * 1024:
            print(f"  [SKIP] Too large (>{MAX_FILE_SIZE_MB} MB): {description}")
            return False

        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)
                if total > MAX_FILE_SIZE_MB * 1024 * 1024:
                    f.close()
                    dest_path.unlink(missing_ok=True)
                    print(f"  [SKIP] Aborted (>{MAX_FILE_SIZE_MB} MB mid-stream): {description}")
                    return False

        size_kb = total / 1024
        print(f"  [OK]   Saved ({size_kb:.0f} KB): {dest_path.name}")
        return True

    except Exception as e:
        print(f"  [FAIL] {description} -- {e}")
        dest_path.unlink(missing_ok=True)
        return False


# -----------------------------------------------------------------
# KAGGLE DOWNLOADER
# -----------------------------------------------------------------
def setup_kaggle_auth():
    """Configure Kaggle authentication from environment or token file."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return KaggleApi
    except ImportError:
        print("[INFO] Kaggle package not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle --quiet")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            return KaggleApi
        except ImportError:
            print("[FAIL] Could not install kaggle. Skipping Kaggle downloads.")
            return None


def download_kaggle_datasets() -> int:
    """Download CSV/PDF files from curated Kaggle datasets. Returns count."""
    KaggleApi = setup_kaggle_auth()
    if KaggleApi is None:
        return 0

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[FAIL] Kaggle authentication failed: {e}")
        print("       Set KAGGLE_API_TOKEN env var or place token in ~/.kaggle/access_token")
        return 0

    print("\n" + "=" * 60)
    print("  KAGGLE DOWNLOADS")
    print("=" * 60)

    downloaded = 0
    temp_dir = DOWNLOAD_DIR / "_kaggle_temp"

    for dataset_slug in KAGGLE_DATASETS:
        if downloaded >= 15:
            break

        dataset_name = dataset_slug.split("/")[-1]
        print(f"\n[{downloaded + 1}/15] Dataset: {dataset_slug}")

        try:
            # Download to temp dir first (Kaggle zips everything)
            temp_dir.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(
                dataset_slug,
                path=str(temp_dir),
                unzip=True,
                quiet=True
            )

            # Find CSV/PDF files in the extracted content
            found_files = []
            for root, dirs, files in os.walk(temp_dir):
                for fname in files:
                    ext = Path(fname).suffix.lower()
                    if ext in ['.csv', '.pdf', '.xlsx']:
                        found_files.append(Path(root) / fname)

            if not found_files:
                print(f"  [SKIP] No CSV/PDF/XLSX files found in {dataset_name}")
                continue

            # Take the first (usually main) file from each dataset
            src_file = found_files[0]
            dest_name = sanitize_filename(f"kaggle_{dataset_name}{src_file.suffix}")

            if file_already_exists(dest_name):
                print(f"  [SKIP] Already exists: {dest_name}")
                downloaded += 1
                continue

            dest_path = DOWNLOAD_DIR / dest_name

            # Move file from temp to download dir
            import shutil
            shutil.move(str(src_file), str(dest_path))

            size_kb = dest_path.stat().st_size / 1024
            print(f"  [OK]   Saved ({size_kb:.0f} KB): {dest_name}")
            downloaded += 1

        except Exception as e:
            print(f"  [FAIL] Error with {dataset_slug}: {e}")

        finally:
            # Clean up temp directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        time.sleep(1)  # Be polite to the API

    print(f"\nKaggle total: {downloaded} files downloaded")
    return downloaded


# -----------------------------------------------------------------
# GOVDATA.DE DOWNLOADER
# -----------------------------------------------------------------
def search_govdata(query: str, rows: int = 5) -> list:
    """Search GovData.de CKAN API for datasets matching a query.
    Returns a list of (title, resource_url, format) tuples."""
    try:
        url = f"{GOVDATA_API_BASE}/package_search"
        params = {"q": query, "rows": rows}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = []
        if data.get("success") and data.get("result", {}).get("results"):
            for dataset in data["result"]["results"]:
                title = dataset.get("title", "unknown")
                for resource in dataset.get("resources", []):
                    fmt = resource.get("format", "").lower()
                    res_url = resource.get("url", "")

                    # Normalize format field (GovData uses EU authority URIs)
                    if "pdf" in fmt:
                        fmt_clean = "pdf"
                    elif "csv" in fmt:
                        fmt_clean = "csv"
                    elif "xlsx" in fmt or "xls" in fmt:
                        fmt_clean = "xlsx"
                    else:
                        continue

                    if res_url and fmt_clean in ALLOWED_FORMATS:
                        results.append((title, res_url, fmt_clean))

        return results

    except Exception as e:
        print(f"  [FAIL] GovData search error for '{query}': {e}")
        return []


def download_govdata_datasets() -> int:
    """Download PDF/CSV files from GovData.de. Returns count."""
    print("\n" + "=" * 60)
    print("  GOVDATA.DE DOWNLOADS")
    print("=" * 60)

    downloaded = 0
    seen_urls = set()  # avoid duplicate downloads

    for query in GOVDATA_SEARCH_QUERIES:
        if downloaded >= 15:
            break

        print(f"\n[{downloaded + 1}/15] Searching: '{query}'")
        results = search_govdata(query, rows=3)

        for title, url, fmt in results:
            if downloaded >= 15:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Build a clean filename
            short_title = sanitize_filename(title)[:80]
            dest_name = f"govdata_{short_title}.{fmt}"

            if file_already_exists(dest_name):
                print(f"  [SKIP] Already exists: {dest_name}")
                downloaded += 1
                continue

            dest_path = DOWNLOAD_DIR / dest_name
            success = download_file(url, dest_path, description=f"{title} (.{fmt})")

            if success:
                downloaded += 1

            time.sleep(0.5)  # Be polite to external servers

    print(f"\nGovData.de total: {downloaded} files downloaded")
    return downloaded


# -----------------------------------------------------------------
# SUMMARY & MAIN
# -----------------------------------------------------------------
def print_summary():
    """Print a summary of all files in the download directory."""
    print("\n" + "=" * 60)
    print("  DOWNLOAD DIRECTORY SUMMARY")
    print("=" * 60)

    files = sorted(DOWNLOAD_DIR.iterdir())
    total_size = 0
    by_ext = {}

    for f in files:
        if f.is_file():
            ext = f.suffix.lower()
            size = f.stat().st_size
            total_size += size
            by_ext[ext] = by_ext.get(ext, 0) + 1

    print(f"   Location: {DOWNLOAD_DIR.resolve()}")
    print(f"   Total files: {sum(by_ext.values())}")
    print(f"   Total size: {total_size / (1024 * 1024):.1f} MB")
    print(f"   Breakdown: {', '.join(f'{ext}: {count}' for ext, count in sorted(by_ext.items()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Download 30 documents from Kaggle & GovData.de for RAG ingestion"
    )
    parser.add_argument("--kaggle-only", action="store_true", help="Download only from Kaggle")
    parser.add_argument("--govdata-only", action="store_true", help="Download only from GovData.de")
    args = parser.parse_args()

    print("RAG Data Scraper -- Kaggle + GovData.de")
    print(f"   Target directory: {DOWNLOAD_DIR.resolve()}")
    ensure_download_dir()

    kaggle_count = 0
    govdata_count = 0

    if not args.govdata_only:
        kaggle_count = download_kaggle_datasets()

    if not args.kaggle_only:
        govdata_count = download_govdata_datasets()

    # Final report
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"   Kaggle:     {kaggle_count} files")
    print(f"   GovData.de: {govdata_count} files")
    print(f"   Total new:  {kaggle_count + govdata_count} files")

    print_summary()


if __name__ == "__main__":
    main()