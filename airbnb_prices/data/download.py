import hashlib
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm


def download_file(
    dest_file: Path, url: str, show_progress: bool = True, chunk_size: int = 1024 * 1024
) -> Path:
    """
    Download the file located at url to dest_file.

    Will follow redirects.
    :param dest_file: The file destination path.
    :param url: A link to the remote file.
    :param show_progress: Print a progress bar to stdout if set to True.
    :param chunk_size: Number of bits to download before saving stream to file.

    ;return: The file the dataset was downloaded to.
    """
    # Ensure the directory we'll put the downloaded file in actually exists
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        file_size = int(response.headers["Content-Length"])
        with open(dest_file, "wb") as db_file:
            with tqdm(
                total=file_size, unit="B", unit_scale=True, disable=not show_progress
            ) as progress_bar:
                while chunk := response.read(chunk_size):
                    db_file.write(chunk)
                    progress_bar.update(chunk_size)
        return dest_file


def checksum(file: Path) -> bytes:
    """
    Compute the md5 checksum of a file.
    """
    hasher = hashlib.md5()
    with open(file, "rb") as open_file:
        while chunk := open_file.read(128 * hasher.block_size):
            hasher.update(chunk)
    return hasher.digest()


def download(dest_file: Path | str = "data/train_airbnb_berlin.csv"):
    dest_file = Path(dest_file)
    correct_checksum = b"\x0f\x85:\xe3\xcdi\x9fRq{\xb7\xa6\xbb\xbda\xc2"
    if not dest_file.exists() or checksum(dest_file) != correct_checksum:
        download_file(dest_file, "http://141.94.23.118:7777/train_airbnb_berlin.csv")
        if checksum(dest_file) != checksum(dest_file):
            raise ValueError("The file was corrupted during download, please try again.")
