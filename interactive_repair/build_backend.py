from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


NAME = "semaphore-tracker"
VERSION = "0.1.0"
PURELIB_TAG = "py3-none-any"


def _normalized_dist_name() -> str:
    return NAME.replace("-", "_").replace(".", "_")


def _wheel_filename() -> str:
    return f"{_normalized_dist_name()}-{VERSION}-{PURELIB_TAG}.whl"


def _dist_info_dir() -> str:
    return f"{_normalized_dist_name()}-{VERSION}.dist-info"


def _metadata() -> str:
    return "\n".join(
        [
            "Metadata-Version: 2.1",
            f"Name: {NAME}",
            f"Version: {VERSION}",
            "Summary: Human-in-the-loop point tracking repair platform",
            "",
        ]
    )


def _wheel() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: semaphore-tracker build_backend",
            "Root-Is-Purelib: true",
            f"Tag: {PURELIB_TAG}",
            "",
        ]
    )


def _hash_bytes(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _write_wheel(wheel_directory: str) -> str:
    wheel_directory = os.fspath(wheel_directory)
    Path(wheel_directory).mkdir(parents=True, exist_ok=True)

    wheel_path = Path(wheel_directory) / _wheel_filename()
    pth_name = f"{_normalized_dist_name()}.pth"
    pth_text = str(Path(__file__).resolve().parent) + os.linesep
    metadata = _metadata().encode("utf-8")
    wheel_text = _wheel().encode("utf-8")
    pth_bytes = pth_text.encode("utf-8")

    record_rows: list[tuple[str, str, str]] = []

    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(pth_name, pth_bytes)
        record_rows.append((pth_name, _hash_bytes(pth_bytes), str(len(pth_bytes))))

        dist_info = _dist_info_dir()
        metadata_name = f"{dist_info}/METADATA"
        wheel_name = f"{dist_info}/WHEEL"
        record_name = f"{dist_info}/RECORD"

        zf.writestr(metadata_name, metadata)
        record_rows.append((metadata_name, _hash_bytes(metadata), str(len(metadata))))

        zf.writestr(wheel_name, wheel_text)
        record_rows.append((wheel_name, _hash_bytes(wheel_text), str(len(wheel_text))))

        record_rows.append((record_name, "", ""))
        record_text = "\n".join(
            f"{path},{hash_value},{size}" for path, hash_value, size in record_rows
        ) + "\n"
        zf.writestr(record_name, record_text.encode("utf-8"))

    return str(wheel_path)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return os.path.basename(_write_wheel(wheel_directory))


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return os.path.basename(_write_wheel(wheel_directory))


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    dist_info = Path(metadata_directory) / _dist_info_dir()
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata(), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel(), encoding="utf-8")
    return dist_info.name


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)
