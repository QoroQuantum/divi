# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Model download and management for divi-ai.

Handles downloading quantised GGUF models from HuggingFace Hub and caching
them locally so they only need to be fetched once.
"""

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "divi-ai"
MODELS_DIR = CACHE_DIR / "models"
CONFIG_FILE = CACHE_DIR / "config.txt"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a downloadable GGUF model."""

    label: str
    repo_id: str
    filename: str
    size_gb: float
    ram_gb: float
    description: str


AVAILABLE_MODELS: dict[str, ModelSpec] = {
    "0.5b": ModelSpec(
        label="0.5B",
        repo_id="Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        filename="qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        size_gb=0.4,
        ram_gb=1.0,
        description="Very fast, basic quality",
    ),
    "1.5b": ModelSpec(
        label="1.5B",
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        size_gb=1.0,
        ram_gb=2.0,
        description="Good balance of quality and speed",
    ),
    "3b": ModelSpec(
        label="3B",
        repo_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        size_gb=1.9,
        ram_gb=3.0,
        description="Better reasoning, moderate speed",
    ),
}

DEFAULT_MODEL_SIZE = "1.5b"

_LOCAL_GGUF_SUBPATH = Path("finetune") / "divi-ai-ft" / "merged_gguf"

# ---------------------------------------------------------------------------
# Local fine-tune detection
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path | None:
    """Walk up from this file to find the git repository root."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return parent
    return None


def find_local_gguf() -> Path | None:
    """Return the path to a locally fine-tuned GGUF, or ``None``.

    Looks for ``*.gguf`` files in ``finetune/divi-ai-ft/merged_gguf/``
    relative to the repository root.  Only finds models in development
    installs where the finetune directory exists.
    """
    repo_root = _find_repo_root()
    if repo_root is None:
        return None
    gguf_dir = repo_root / _LOCAL_GGUF_SUBPATH
    if gguf_dir.is_dir():
        ggufs = sorted(gguf_dir.glob("*.gguf"))
        if ggufs:
            return ggufs[0]
    return None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def ensure_model(size: str) -> Path:
    """Return the local path to the GGUF file, downloading if necessary.

    Parameters
    ----------
    size:
        One of ``"0.5b"``, ``"1.5b"``, ``"3b"``.

    Returns
    -------
    Path
        Absolute path to the cached ``.gguf`` file.

    Raises
    ------
    KeyError
        If *size* is not a recognised model key.
    """
    spec = AVAILABLE_MODELS[size]
    local_dir = MODELS_DIR / size
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / spec.filename
    if local_path.exists():
        return local_path

    print(f"Downloading {spec.label} model ({spec.size_gb} GB) …")
    downloaded = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        local_dir=str(local_dir),
    )
    return Path(downloaded)


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------


def select_model_interactive() -> str:
    """Show an interactive prompt for the user to pick a model size.

    Returns
    -------
    str
        The chosen model key (e.g. ``"1.5b"``).
    """
    console = Console()

    table = Table(title="Available Models", show_lines=True)
    table.add_column("Key", style="bold cyan")
    table.add_column("Size")
    table.add_column("Download")
    table.add_column("RAM")
    table.add_column("Notes", style="dim")

    for key, spec in AVAILABLE_MODELS.items():
        table.add_row(
            key,
            spec.label,
            f"{spec.size_gb:.1f} GB",
            f"~{spec.ram_gb:.0f} GB",
            spec.description,
        )

    console.print()
    console.print(table)

    choice = Prompt.ask(
        "Choose model size",
        choices=list(AVAILABLE_MODELS),
        default=DEFAULT_MODEL_SIZE,
    )
    save_preferred_model(choice)
    return choice


def save_preferred_model(size: str) -> None:
    """Persist the user's model preference to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(size)


def load_preferred_model() -> str | None:
    """Load a previously saved model preference, or ``None``."""
    if CONFIG_FILE.exists():
        saved = CONFIG_FILE.read_text().strip()
        if saved in AVAILABLE_MODELS:
            return saved
    return None
