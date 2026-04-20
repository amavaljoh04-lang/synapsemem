"""CrossCodeEval dataset bootstrap for the UI-launched harness.

The official tar.xz (~40 MB) is downloaded and expanded **once** into
``/app/data/cce`` on first use. Subsequent runs reuse the cached tree.
No network access is required after the first bootstrap, so the bench
is offline-repeatable — important because the whole product runs on a
single GPU box with inconsistent outbound connectivity.

Language subsets available in the release:
``python``, ``java``, ``typescript``, ``csharp``.

Variants per language:

* ``line_completion.jsonl`` — plain (no crossfile).
* ``line_completion_oracle_bm25.jsonl`` — oracle cross-file blob.
* ``line_completion_rg1_bm25.jsonl`` — retrieved cross-file blob.
* Similar ``*_openai_cosine_sim.jsonl`` / ``*_unixcoder_cosine_sim.jsonl``
  variants shipped by the CCE authors.

For SynapseMem we default to the oracle blob: it's the strongest
ceiling for cross-file context, so it's the cleanest signal of whether
the memory layer is extracting anything useful.
"""

from __future__ import annotations

import lzma
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx

from ..config import settings

CCE_URL = "https://github.com/amazon-science/cceval/raw/main/data/crosscodeeval_data.tar.xz"

DATASET_VARIANTS: tuple[str, ...] = (
    "line_completion_oracle_bm25",
    "line_completion_rg1_bm25",
    "line_completion_oracle_openai_cosine_sim",
    "line_completion_rg1_openai_cosine_sim",
    "line_completion_oracle_unixcoder_cosine_sim",
    "line_completion_rg1_unixcoder_cosine_sim",
    "line_completion",
)

DEFAULT_LANGUAGE = "python"
DEFAULT_VARIANT = "line_completion_oracle_bm25"


def _root() -> Path:
    """Where we unpack the dataset on disk.

    Lives under ``settings.data_dir`` so it survives container
    restarts (data dir is volume-mounted in docker-compose).
    """
    return Path(settings.data_dir) / "cce"


def dataset_path(language: str, variant: str) -> Path:
    """Return the on-disk path for a ``(language, variant)`` pair.

    No side effects — caller decides whether to ensure / download.
    """
    return _root() / language / f"{variant}.jsonl"


def is_installed() -> bool:
    """True iff the default Python/oracle file is already on disk."""
    return dataset_path(DEFAULT_LANGUAGE, DEFAULT_VARIANT).is_file()


def available_languages() -> list[str]:
    """Language subsets that have already been unpacked."""
    root = _root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def available_variants(language: str) -> list[str]:
    """Variants present on disk for ``language``."""
    lang_dir = _root() / language
    if not lang_dir.is_dir():
        return []
    return sorted(p.stem for p in lang_dir.glob("*.jsonl"))


async def ensure_installed() -> Path:
    """Download + extract the CCE release if it's not on disk already.

    Returns the path to the default JSONL so callers can chain into
    ``load_samples``. Idempotent: if the file is already there, no
    network call is made.
    """
    default = dataset_path(DEFAULT_LANGUAGE, DEFAULT_VARIANT)
    if default.is_file():
        return default
    _root().mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.xz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        async with (
            httpx.AsyncClient(timeout=600.0, follow_redirects=True) as http,
            http.stream("GET", CCE_URL) as r,
        ):
            r.raise_for_status()
            with open(tmp_path, "wb") as fh:
                async for chunk in r.aiter_bytes():
                    fh.write(chunk)
        _extract_tar_xz(tmp_path, _root())
    finally:
        tmp_path.unlink(missing_ok=True)
    if not default.is_file():
        raise RuntimeError(
            "CCE tar.xz extracted but expected default JSONL is missing: "
            f"{default}"
        )
    return default


def _extract_tar_xz(archive: Path, dest: Path) -> None:
    """Decompress + untar ``archive`` into ``dest``.

    We do it in two steps (lzma → plain tar) so very old tarfile
    implementations that lack native xz support still work. Paths
    containing ``..`` are rejected to avoid zip-slip.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_tar = Path(tmp.name)
    try:
        with lzma.open(archive, "rb") as src, open(tmp_tar, "wb") as out:
            shutil.copyfileobj(src, out)
        with tarfile.open(tmp_tar, "r:") as tar:
            for member in tar.getmembers():
                norm = Path(member.name).as_posix()
                if norm.startswith("/") or ".." in norm.split("/"):
                    raise RuntimeError(f"unsafe path in tar: {member.name}")
            tar.extractall(dest)
    finally:
        tmp_tar.unlink(missing_ok=True)
