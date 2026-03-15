# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Model download and management for divi-ai.

Handles downloading quantised GGUF models from HuggingFace Hub and caching
them locally so they only need to be fetched once.
"""

import io
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
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
    "1.5b": ModelSpec(
        label="1.5B",
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        size_gb=1.0,
        ram_gb=2.0,
        description="Basic quality, fastest",
    ),
    "3b": ModelSpec(
        label="3B",
        repo_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        size_gb=1.9,
        ram_gb=3.0,
        description="Balanced quality and speed",
    ),
    "7b": ModelSpec(
        label="7B",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        size_gb=4.5,
        ram_gb=6.0,
        description="Best quality, needs 6 GB+ RAM",
    ),
}

DEFAULT_MODEL_SIZE = "7b"

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def ensure_model(size: str) -> Path:
    """Return the local path to the GGUF file, downloading if necessary.

    Parameters
    ----------
    size:
        One of ``"1.5b"``, ``"3b"``, ``"7b"``.

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


# ---------------------------------------------------------------------------
# LLM loading (shared by CLI and eval)
# ---------------------------------------------------------------------------

_console = Console()


def load_llm(model_path: Path, *, debug: bool = False) -> Llama:
    """Load a GGUF model via llama-cpp-python.

    Parameters
    ----------
    model_path:
        Path to the ``.gguf`` file.
    debug:
        If ``True``, print loading messages and keep llama.cpp stderr.

    Returns
    -------
    Llama
        A loaded ``llama_cpp.Llama`` instance ready for inference.
    """
    n_ctx = 16384 if "7b" in model_path.stem.lower() else 8192

    if debug:
        _console.print(f"[dim]Loading model from {model_path} …[/dim]")
        return Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=0,
            verbose=False,
        )
    with redirect_stderr(io.StringIO()):
        return Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=0,
            verbose=False,
        )
