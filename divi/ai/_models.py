# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Model download and management for divi-ai.

Handles downloading quantised GGUF models from HuggingFace Hub and caching
them locally so they only need to be fetched once.  Model metadata (file
sizes) is fetched from HuggingFace on first use and cached to disk; hardcoded
defaults are used as a last-resort fallback when offline.
"""

import io
import json
import logging
import sys
from contextlib import redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from llama_cpp import Llama
from platformdirs import user_cache_dir
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from ._system import detect_arch, detect_ram_gb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

CACHE_DIR = Path(user_cache_dir("divi-ai"))
MODELS_DIR = CACHE_DIR / "models"
CONFIG_FILE = CACHE_DIR / "config.txt"
METADATA_CACHE_FILE = CACHE_DIR / "model_metadata.json"

# RAM multiplier: approximate runtime memory ≈ GGUF file size × this factor.
_RAM_MULTIPLIER = 1.2


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a downloadable GGUF model."""

    label: str
    repo_id: str
    filename: str
    n_ctx: int
    default_size_gb: float


AVAILABLE_MODELS: dict[str, ModelSpec] = {
    "1.5b": ModelSpec(
        label="Qwen 2.5 Coder 1.5B",
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        n_ctx=8192,
        default_size_gb=1.0,
    ),
    "3b": ModelSpec(
        label="Qwen 2.5 Coder 3B",
        repo_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        n_ctx=8192,
        default_size_gb=1.9,
    ),
    "7b": ModelSpec(
        label="Qwen 2.5 Coder 7B",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        n_ctx=16384,
        default_size_gb=4.5,
    ),
    "14b": ModelSpec(
        label="Qwen 2.5 Coder 14B",
        repo_id="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        n_ctx=16384,
        default_size_gb=8.4,
    ),
    "e2b": ModelSpec(
        label="Gemma 4 E2B",
        repo_id="unsloth/gemma-4-E2B-it-GGUF",
        filename="gemma-4-E2B-it-Q4_K_M.gguf",
        n_ctx=8192,
        default_size_gb=2.9,
    ),
    "e4b": ModelSpec(
        label="Gemma 4 E4B",
        repo_id="unsloth/gemma-4-E4B-it-GGUF",
        filename="gemma-4-E4B-it-Q4_K_M.gguf",
        n_ctx=8192,
        default_size_gb=4.6,
    ),
}

DEFAULT_MODEL_SIZE = "7b"

# ---------------------------------------------------------------------------
# HuggingFace metadata cache
# ---------------------------------------------------------------------------


def _load_metadata_cache() -> dict:
    """Load the cached model metadata from disk."""
    try:
        return json.loads(METADATA_CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_metadata_cache(cache: dict) -> None:
    """Persist model metadata cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _fetch_model_size_from_hf(spec: ModelSpec) -> float | None:
    """Fetch the GGUF file size from HuggingFace (in GB).

    Returns ``None`` on any failure (network, timeout, file not found).
    """
    try:
        api = HfApi()
        info = api.model_info(spec.repo_id, files_metadata=True, timeout=10)
        for f in info.siblings:
            if f.rfilename == spec.filename and f.size is not None:
                return f.size / (1024**3)
    except Exception:
        logger.debug("Failed to fetch metadata for %s", spec.repo_id, exc_info=True)
    return None


def resolve_model_sizes(models: dict[str, ModelSpec]) -> dict[str, float]:
    """Resolve download sizes for all models.

    Strategy: cache → HuggingFace API → hardcoded fallback.

    Returns
    -------
    dict[str, float]
        Mapping of model key to download size in GB.
    """
    cache = _load_metadata_cache()
    sizes: dict[str, float] = {}
    updated = False

    for key, spec in models.items():
        # 1. Try cache
        if key in cache:
            sizes[key] = cache[key]["size_gb"]
            continue

        # 2. Try HuggingFace
        hf_size = _fetch_model_size_from_hf(spec)
        if hf_size is not None:
            sizes[key] = hf_size
            cache[key] = {
                "size_gb": round(hf_size, 2),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            updated = True
            continue

        # 3. Hardcoded fallback
        sizes[key] = spec.default_size_gb

    if updated:
        _save_metadata_cache(cache)

    return sizes


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def ensure_model(size: str) -> Path:
    """Return the local path to the GGUF file, downloading if necessary.

    Parameters
    ----------
    size:
        One of the keys in :data:`AVAILABLE_MODELS`.

    Returns
    -------
    Path
        Absolute path to the cached ``.gguf`` file.

    Raises
    ------
    KeyError
        If *size* is not a recognized model key.
    """
    spec = AVAILABLE_MODELS[size]
    local_dir = MODELS_DIR / size
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / spec.filename
    if local_path.exists():
        return local_path

    print(f"Downloading {spec.label} model …")
    downloaded = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        local_dir=str(local_dir),
    )
    return Path(downloaded)


# ---------------------------------------------------------------------------
# Model recommendations
# ---------------------------------------------------------------------------


def get_recommended_models(arch: str, ram_gb: float | None) -> set[str]:
    """Return the set of model keys that fit the detected system.

    Recommendations are based on system specs (architecture and available RAM),
    not on model quality or performance.
    """
    if ram_gb is None:
        return set()

    is_apple = arch == "apple_silicon"

    if is_apple and ram_gb >= 16.0:
        return {"7b", "14b"}
    if is_apple:
        return {"1.5b", "3b", "e2b", "e4b", "7b"}
    if ram_gb >= 32.0:
        return {"7b", "14b"}
    if ram_gb >= 16.0:
        return {"e4b", "7b", "14b"}
    return {"1.5b", "3b", "e2b"}


# ---------------------------------------------------------------------------
# Arrow-key menu
# ---------------------------------------------------------------------------


def _read_key() -> str | None:
    """Read a single keypress and return ``'up'``, ``'down'``, ``'enter'``, ``'escape'``, or ``None``."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch == "\r":
            return "enter"
        if ch == "\x1b":
            return "escape"
        if ch == "\x03":
            return "escape"
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "up"
            if ch2 == "P":
                return "down"
        return None
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                return "enter"
            if ch == "\x03":
                return "escape"
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    if ch3 == "B":
                        return "down"
                return "escape"
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------


def _build_table(
    sizes: dict[str, float],
    recommended: set[str],
    selected_index: int,
) -> Table:
    """Build the model table with the selected row highlighted."""
    table = Table(title="Available Models", show_lines=True)
    table.add_column("Model")
    table.add_column("Download")
    table.add_column("Est. RAM Needed")
    table.add_column("")

    for i, (key, spec) in enumerate(AVAILABLE_MODELS.items()):
        size_gb = sizes[key]
        ram_est = size_gb * _RAM_MULTIPLIER
        marker = "[bold green]✓ Recommended[/bold green]" if key in recommended else ""
        prefix = "❯ " if i == selected_index else "  "
        style = "bold" if i == selected_index else None
        table.add_row(
            prefix + spec.label,
            f"{size_gb:.1f} GB",
            f"~{ram_est:.0f} GB",
            marker,
            style=style,
        )

    return table


def select_model_interactive() -> str:
    """Show an interactive prompt for the user to pick a model size.

    Displays detected system information and an interactive table of
    available models navigated with arrow keys.

    Returns
    -------
    str
        The chosen model key (e.g. ``"e2b"``).
    """
    console = Console()

    # System info
    arch = detect_arch()
    ram_gb = detect_ram_gb()

    # Resolve sizes (cache → HF → fallback)
    sizes = resolve_model_sizes(AVAILABLE_MODELS)
    recommended = get_recommended_models(arch, ram_gb)
    model_keys = list(AVAILABLE_MODELS.keys())

    # Default to the largest recommended model, or fall back to global default
    if recommended:
        default = max(recommended, key=lambda k: model_keys.index(k))
    else:
        default = DEFAULT_MODEL_SIZE
    current = model_keys.index(default)

    def _render():
        parts = []
        if ram_gb is not None:
            parts.append(
                Align.center(f"[dim]System: {arch}, {ram_gb:.1f} GB RAM[/dim]\n")
            )
        else:
            parts.append(Align.center(f"[dim]System: {arch}[/dim]\n"))
        parts.append(Align.center(_build_table(sizes, recommended, current)))
        if recommended:
            parts.append(
                Align.center(
                    "[dim]✓ = recommended for your system specs, not a performance ranking[/dim]"
                )
            )
        parts.append(
            Align.center("[dim]↑↓ to select, Enter to confirm, Esc to cancel[/dim]")
        )
        return Group(*parts)

    with Live(_render(), console=console, auto_refresh=False, transient=True) as live:
        while True:
            key = _read_key()
            if key == "up" and current > 0:
                current -= 1
            elif key == "down" and current < len(model_keys) - 1:
                current += 1
            elif key == "enter":
                break
            elif key == "escape":
                console.print("[dim]Cancelled.[/dim]")
                raise SystemExit(0)
            live.update(_render())
            live.refresh()

    choice = model_keys[current]
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


def load_llm(model_path: Path, *, n_ctx: int = 8192, debug: bool = False) -> Llama:
    """Load a GGUF model via llama-cpp-python.

    Parameters
    ----------
    model_path:
        Path to the ``.gguf`` file.
    n_ctx:
        Context window size in tokens.
    debug:
        If ``True``, print loading messages and keep llama.cpp stderr.

    Returns
    -------
    Llama
        A loaded ``llama_cpp.Llama`` instance ready for inference.
    """
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
