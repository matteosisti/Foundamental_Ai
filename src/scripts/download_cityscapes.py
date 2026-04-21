# src/scripts/download_cityscapes.py
#
# Downloads Cityscapes validation split directly to Colab local disk.
# Requires CITYSCAPES_USER and CITYSCAPES_PASS set as Colab secrets.
# Output: /content/cityscapes/leftImg8bit/val and /content/cityscapes/gtFine/val
#
# Usage (in Colab):
#   from google.colab import userdata
#   import subprocess
#   subprocess.run(["python3", "-m", "src.scripts.download_cityscapes"])

import os
import subprocess
import zipfile
from pathlib import Path


COOKIES   = "/tmp/cityscapes_cookies.txt"
OUT_DIR   = "/content/cityscapes"
TMP_DIR   = "/tmp"

PACKAGE_IDS = {
    "gtFine":       3,   # ~241 MB — ground truth labels
    "leftImg8bit":  9,   # ~1.5 GB — RGB images (train+val+test)
}


def login(username: str, password: str) -> None:
    print("[cityscapes] logging in...")
    subprocess.run([
        "wget", "--keep-session-cookies",
        f"--save-cookies={COOKIES}",
        "--post-data",
        f"username={username}&password={password}&submit=Login",
        "https://www.cityscapes-dataset.com/login/",
        "-q", "-O", "/tmp/login.html",
    ], check=True)
    print("[cityscapes] login complete")


def download_package(name: str, package_id: int) -> str:
    out_path = f"{TMP_DIR}/{name}.zip"
    print(f"[cityscapes] downloading {name} (packageID={package_id})...")
    subprocess.run([
        "wget", f"--load-cookies={COOKIES}",
        f"https://www.cityscapes-dataset.com/file-handling/?packageID={package_id}",
        "-O", out_path, "-q", "--show-progress",
    ], check=True)
    print(f"[cityscapes] downloaded {out_path}")
    return out_path


def extract_val_only(zip_path: str, out_dir: str) -> None:
    print(f"[cityscapes] extracting val split from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        val_members = [m for m in z.namelist() if "/val/" in m]
        z.extractall(out_dir, members=val_members)
    print(f"[cityscapes] extracted {len(val_members)} files to {out_dir}")


def main():
    try:
        from google.colab import userdata
        username = userdata.get("CITYSCAPES_USER")
        password = userdata.get("CITYSCAPES_PASS")
    except Exception:
        username = os.environ.get("CITYSCAPES_USER")
        password = os.environ.get("CITYSCAPES_PASS")

    if not username or not password:
        raise ValueError(
            "CITYSCAPES_USER and CITYSCAPES_PASS must be set "
            "as Colab secrets or environment variables."
        )

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    login(username, password)

    for name, pkg_id in PACKAGE_IDS.items():
        zip_path = download_package(name, pkg_id)
        extract_val_only(zip_path, OUT_DIR)
        os.remove(zip_path)
        print(f"[cityscapes] cleaned up {zip_path}")

    # Verify
    n_imgs = len(list(Path(f"{OUT_DIR}/leftImg8bit/val").rglob("*.png")))
    n_gts  = len(list(Path(f"{OUT_DIR}/gtFine/val").rglob("*labelIds.png")))
    print(f"\n[cityscapes] val set ready:")
    print(f"  images: {n_imgs}")
    print(f"  labels: {n_gts}")
    print(f"  path:   {OUT_DIR}")


if __name__ == "__main__":
    main()
