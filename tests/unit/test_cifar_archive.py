import io
import tarfile
from pathlib import Path

import pytest

from robustvision.data.cifar_archive import extract_archive, is_valid_archive, md5sum


def make_archive(path: Path, member: str = "cifar-10-batches-py/data_batch_1"):
    with tarfile.open(path, "w:gz") as archive:
        content = b"fixture"
        info = tarfile.TarInfo(member)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))


def test_valid_local_archive_extracts_once_without_overwrite(tmp_path: Path):
    archive = tmp_path / "cifar-10-python.tar.gz"
    make_archive(archive)
    assert is_valid_archive(archive, md5sum(archive))
    target = extract_archive(archive, tmp_path)
    assert (target / "data_batch_1").read_bytes() == b"fixture"
    assert extract_archive(archive, tmp_path) == target


def test_invalid_checksum_is_rejected(tmp_path: Path):
    archive = tmp_path / "cifar-10-python.tar.gz"
    make_archive(archive)
    assert not is_valid_archive(archive, "0" * 32)


def test_path_traversal_member_is_rejected(tmp_path: Path):
    archive = tmp_path / "unsafe.tar.gz"
    make_archive(archive, "../outside.txt")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, tmp_path / "data")
