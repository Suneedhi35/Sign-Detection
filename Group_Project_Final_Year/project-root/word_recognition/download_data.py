"""
Download ISL Dictionary videos from Google Drive.
==================================================
Reads folder URLs from ISL_Dictionary_words.csv and downloads
each letter/number folder's contents into data/raw_videos/<folder_name>/

Usage:
    python -m word_recognition.download_data
    python -m word_recognition.download_data --letters A B C
    python -m word_recognition.download_data --skip-existing
"""

import os
import sys
import argparse

try:
    import gdown
except ImportError:
    print("ERROR: gdown is required.  Install with:  pip install gdown>=5.1.0")
    sys.exit(1)

from .config import Config


def _extract_folder_id(url: str) -> str:
    """Extract the Google Drive folder ID from a full URL."""
    # URL forms:
    #   https://drive.google.com/drive/folders/<ID>
    #   https://drive.google.com/drive/folders/<ID>?usp=drive_link
    parts = url.rstrip("/").split("/")
    raw_id = parts[-1]
    if "?" in raw_id:
        raw_id = raw_id.split("?")[0]
    return raw_id


def download_folder(name: str, url: str, skip_existing: bool = True) -> int:
    """Download a single Drive folder. Returns count of downloaded files."""
    target_dir = os.path.join(Config.RAW_VIDEO_DIR, name)
    os.makedirs(target_dir, exist_ok=True)

    if skip_existing and os.listdir(target_dir):
        print(f"  ⏭️  Skipping '{name}' — already has {len(os.listdir(target_dir))} files")
        return 0

    print(f"\n📥 Downloading folder: {name}")
    print(f"   URL: {url}")
    print(f"   Target: {target_dir}")

    try:
        gdown.download_folder(
            url=url,
            output=target_dir,
            quiet=False,
            use_cookies=False,
        )
        count = len([
            f for f in os.listdir(target_dir)
            if os.path.isfile(os.path.join(target_dir, f))
        ])
        print(f"  ✅ Downloaded {count} files for '{name}'")
        return count
    except Exception as e:
        print(f"  ❌ Failed to download '{name}': {e}")
        print(f"     You can manually download from: {url}")
        print(f"     Place files in: {target_dir}")
        return 0


def download_all(letters: list[str] | None = None, skip_existing: bool = True):
    """Download all (or specified) ISL dictionary video folders."""
    Config.ensure_dirs()

    if not Config.DRIVE_FOLDERS:
        Config.load_drive_links()

    if not Config.DRIVE_FOLDERS:
        print("❌ No Drive folder links found. Check ISL_Dictionary_words.csv")
        return

    folders = Config.DRIVE_FOLDERS
    if letters:
        letters_upper = [l.upper() for l in letters]
        folders = {k: v for k, v in folders.items() if k.upper() in letters_upper}
        if not folders:
            print(f"❌ No matching folders for: {letters}")
            return

    print("=" * 60)
    print("  ISL Dictionary Video Downloader")
    print("=" * 60)
    print(f"  Folders to download: {len(folders)}")
    print(f"  Target directory:    {Config.RAW_VIDEO_DIR}")
    print(f"  Skip existing:       {skip_existing}")
    print("=" * 60)

    total_files = 0
    for name, url in sorted(folders.items()):
        count = download_folder(name, url, skip_existing)
        total_files += count

    print("\n" + "=" * 60)
    print(f"  ✅ Done! Total new files downloaded: {total_files}")
    print(f"  📂 Videos saved in: {Config.RAW_VIDEO_DIR}")
    print("=" * 60)

    # Show summary of what we have
    print("\n📊 Current dataset summary:")
    if os.path.exists(Config.RAW_VIDEO_DIR):
        for folder in sorted(os.listdir(Config.RAW_VIDEO_DIR)):
            folder_path = os.path.join(Config.RAW_VIDEO_DIR, folder)
            if os.path.isdir(folder_path):
                n = len([f for f in os.listdir(folder_path)
                         if os.path.isfile(os.path.join(folder_path, f))])
                print(f"   {folder:20s} → {n} videos")


def main():
    parser = argparse.ArgumentParser(
        description="Download ISL Dictionary videos from Google Drive"
    )
    parser.add_argument(
        "--letters", nargs="*", default=None,
        help="Specific letter folders to download (e.g., A B C). Default: all."
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip folders that already have files (default: True)."
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-download even if folder already has files."
    )
    args = parser.parse_args()

    skip = not args.force and args.skip_existing
    download_all(letters=args.letters, skip_existing=skip)


if __name__ == "__main__":
    main()
