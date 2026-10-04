"""Fetch and safely unpack the organizer's archive outside tracked source files."""
from pathlib import Path
import hashlib
import os

import requests
import libarchive

URL = "https://www.gtec.at/downloads_QyTs23/Hackathon/stroke-rehab.rar"
SHA256 = "79c3719513b7ecf5219cb9f78efc34801d6a893df00dd189b7cdc0bc5eb76fde"
MAX_ARCHIVE_BYTES = 500_000_000
ALLOWED_NAMES = {f"P{p}_{stage}_{run}.mat"
                 for p in (1, 2, 3) for stage in ("pre", "post")
                 for run in ("training", "test")}
# Supplied documents are useful, but the Windows thumbnail cache is not.
ALLOWED_NAMES |= {"DatasetInformation.pdf", "overview.pdf", "StrokeRehab.pdf",
                  "Gruenwald et al. - 2019 - Time-Variant Linear Discriminant Analysis Improves.pdf",
                  "montage.png"}


def download_and_extract(archive_path="data/stroke-rehab.rar",
                         destination="data/stroke-rehab"):
    archive_path = Path(archive_path)
    destination = Path(destination)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.is_file():
        temp = archive_path.with_name(archive_path.name + ".partial")
        digest = hashlib.sha256()
        size = 0
        try:
            with requests.get(URL, stream=True, timeout=90) as response:
                response.raise_for_status()
                with temp.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            size += len(chunk)
                            if size > MAX_ARCHIVE_BYTES:
                                raise ValueError("Archive exceeds download limit")
                            digest.update(chunk)
                            output.write(chunk)
            if digest.hexdigest() != SHA256:
                raise ValueError("Organizer archive digest mismatch")
            os.replace(temp, archive_path)
        finally:
            temp.unlink(missing_ok=True)
    else:
        with archive_path.open("rb") as stream:
            verified = hashlib.file_digest(stream, "sha256").hexdigest() == SHA256
        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES or not verified:
            raise ValueError("Cached archive is too large or has an unexpected SHA-256")

    destination.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with libarchive.file_reader(str(archive_path)) as entries:
        for entry in entries:
            if entry.pathname == "stroke-rehab" and entry.filetype == 16384:
                continue
            # The source archive includes a Windows thumbnail cache; intentionally skip it.
            if entry.pathname == "stroke-rehab/Thumbs.db" and entry.filetype == 32768:
                continue
            # Reject path traversal, unexpected files, links and duplicates.
            name = Path(entry.pathname)
            if (len(name.parts) != 2 or name.parts[0] != "stroke-rehab"
                    or name.name not in ALLOWED_NAMES or entry.filetype != 32768
                    or entry.size > 40_000_000 or name.name in extracted):
                raise ValueError(f"Unexpected archive entry: {entry.pathname!r}")
            extracted.add(name.name)
            target = destination / name.name
            with target.open("wb") as output:
                for chunk in entry.get_blocks():
                    output.write(chunk)
    if not {name for name in ALLOWED_NAMES if name.endswith(".mat")} <= extracted:
        raise ValueError("Archive lacks one or more EEG sessions")
    print(f"Verified SHA-256 {SHA256}; extracted {len(extracted)} data/document files "
          f"to {destination}")
