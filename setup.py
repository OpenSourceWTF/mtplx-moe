"""Release-build commands for deterministic source distributions."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import tarfile
from typing import Any

from setuptools import setup
from setuptools.command.sdist import sdist


class ReproducibleSdist(sdist):
    """Canonicalize gzip and tar metadata when SOURCE_DATE_EPOCH is set."""

    @staticmethod
    def _source_date_epoch() -> int | None:
        value = os.environ.get("SOURCE_DATE_EPOCH")
        if value is None:
            return None
        epoch = int(value)
        if epoch < 0:
            raise ValueError("SOURCE_DATE_EPOCH must be non-negative")
        return epoch

    def make_archive(
        self,
        base_name: str,
        format: str,
        root_dir: str | None = None,
        base_dir: str | None = None,
        **kwargs: Any,
    ) -> str:
        epoch = self._source_date_epoch()
        if epoch is None or format != "gztar":
            return super().make_archive(
                base_name,
                format,
                root_dir=root_dir,
                base_dir=base_dir,
                **kwargs,
            )

        archive_path = Path(f"{base_name}.tar.gz")
        source_root = Path(root_dir) if root_dir is not None else Path.cwd()
        source = source_root / (base_dir or ".")
        arcname = base_dir or source.name

        def canonical_header(info: tarfile.TarInfo) -> tarfile.TarInfo:
            info.mtime = epoch
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            return info

        with archive_path.open("wb") as raw_archive:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_archive,
                mtime=epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    archive.add(
                        source,
                        arcname=arcname,
                        recursive=True,
                        filter=canonical_header,
                    )
        return str(archive_path)


setup(cmdclass={"sdist": ReproducibleSdist})
