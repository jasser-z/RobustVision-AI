"""Safe, offline-first CIFAR-10 archive installation."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

OFFICIAL_CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXTRACTED_DIRECTORY = "cifar-10-batches-py"


def md5sum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 -- required by CIFAR-10's published integrity checksum.
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_archive(path: str | Path, expected_md5: str = OFFICIAL_CIFAR10_MD5) -> bool:
    path = Path(path)
    return path.is_file() and md5sum(path).lower() == expected_md5.lower()


def _safe_members(archive: tarfile.TarFile):
    for member in archive.getmembers():
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsupported archive member type: {member.name}")
        yield member


def extract_archive(archive_path: str | Path, data_dir: str | Path) -> Path:
    """Extract a verified archive atomically enough to avoid partial target directories.

    Existing target data is preserved. Torchvision performs the final per-file integrity
    verification after this function returns.
    """
    archive_path, data_dir = Path(archive_path), Path(data_dir)
    target = data_dir / EXTRACTED_DIRECTORY
    if target.is_dir():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cifar10-extract-", dir=data_dir))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _safe_members(archive)
            if sys.version_info >= (3, 12):
                archive.extractall(staging, members=members, filter="data")
            else:  # Python 3.10/3.11: members were already checked by _safe_members.
                archive.extractall(staging, members=members)
        extracted = staging / EXTRACTED_DIRECTORY
        if not extracted.is_dir():
            raise ValueError(f"Archive does not contain {EXTRACTED_DIRECTORY}")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing CIFAR target: {target}")
        os.replace(extracted, target)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_local_archive(data_dir: str | Path) -> Path | None:
    """Install `data/downloads/cifar-10-python.tar.gz` only when its official MD5 matches."""
    data_dir = Path(data_dir)
    archive = data_dir / "downloads" / ARCHIVE_NAME
    if not is_valid_archive(archive):
        return None
    return extract_archive(archive, data_dir)
