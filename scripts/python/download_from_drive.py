"""Download GEE export CSVs from Google Drive to data/raw/gee/.

First run will open a browser for OAuth consent (one-time).
"""

import io
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "gee"
TOKEN_FILE = Path.home() / ".config" / "earthengine" / "drive_token.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

FOLDERS = [
    "kcact_henan_training_data",
    "kcact_shandong_training_data",
    "kcact_anhui_training_data",
]


def get_creds():
    """Authenticate with Google Drive, reusing token if valid."""
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json())
        return creds

    # OAuth desktop app flow — opens browser for consent
    flow = InstalledAppFlow.from_client_secrets_file(
        str(Path.home() / ".config" / "earthengine" / "credentials"), SCOPES
    )
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json())
    return creds


def find_folder(service, name):
    results = (
        service.files()
        .list(
            q=f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            fields="files(id, name)",
            pageSize=10,
        )
        .execute()
    )
    folders = results.get("files", [])
    return folders[0]["id"] if folders else None


def download_folder(service, folder_name, dest_dir):
    folder_id = find_folder(service, folder_name)
    if not folder_id:
        print(f"  [WARN] Folder not found: {folder_name}")
        return 0

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = []
    page_token = None
    while True:
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, size)",
                pageSize=100,
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken")
        if not page_token:
            break

    count = 0
    for f in files:
        dest_path = dest_dir / f["name"]
        if dest_path.exists() and dest_path.stat().st_size > 100:
            print(f"    [SKIP] {f['name']} (exists)")
            count += 1
            continue

        req = service.files().get_media(fileId=f["id"])
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        dest_path.write_bytes(buf.read())
        print(f"    [OK] {f['name']} ({f.get('size', '?')} bytes)")
        count += 1

    return count


def main():
    print("Authenticating with Google Drive...")
    creds = get_creds()
    service = build("drive", "v3", credentials=creds)

    total = 0
    for folder_name in FOLDERS:
        print(f"\nDownloading: {folder_name}")
        n = download_folder(service, folder_name, DATA_DIR)
        total += n
        print(f"  {n} files")

    print(f"\nDone. {total} files saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
