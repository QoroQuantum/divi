# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Document chunking, embedding, and FAISS index management.

Two workflows:

1. **CI / developer** — ``build_index()`` scans the local repo, chunks and
   embeds the docs, and writes ``vectors.faiss`` + ``chunks.json`` to an
   output directory.  These files are then published to HuggingFace Hub.
2. **End-user** — ``load_index()`` reads the pre-built files shipped
   in the ``divi/ai/_data/`` package directory.

Chunking strategies:
- ``.py`` files are parsed with :mod:`ast` to extract function / class
  docstrings with their signatures (structured, noise-free).
- ``.rst`` / ``.md`` files use character-based chunking at ~1024 chars.
- Every chunk is prefixed with its source path for embedding context.
"""

import ast
import json
import os
import re
import subprocess
import tomllib
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from docutils.frontend import OptionParser
from docutils.nodes import section, system_message, title
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from fastembed import TextEmbedding
from rich.console import Console
from rich.progress import track

from ._types import ChunkMeta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
CHUNK_SIZE = 1024  # characters for docs (approximate token proxy)
CHUNK_OVERLAP = 128
MIN_CHUNK_LENGTH = 50  # skip chunks shorter than this

EXTENSIONS = {".py", ".rst", ".md"}

# Files and directories to exclude from indexing — they add noise without
# providing useful user-facing content.
SKIP_NAMES = {
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
}
SKIP_PREFIXES = ("test_", "conftest")
INCLUDE_DIRS = {"divi", "docs", "tutorials", "tests"}


# Sections to skip — they contain navigation links but no substance
SKIP_SECTIONS = {
    "see also",
    "next steps",
    "references",
    "contents",
    "table of contents",
    "indices and tables",
}

# Markdown: # Header
_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
# TOML: [section] or [[array-of-tables]]
_TOML_SECTION_RE = re.compile(r"^\[{1,2}([^\]]+)\]{1,2}\s*$")
# Docutils noise: system messages + role-resolution chatter
_DOCUTILS_NOISE_RE = re.compile(
    r"(^/.+\.rst:\d+: \(.+\) .+$)|(^Trying \".*\" as canonical role name\.\s*$)",
    re.MULTILINE,
)
# Chunk prefix lines added for display: [Source: ...] or [Module: ...]
_EMBED_PREFIX_RE = re.compile(r"^\[(Source|Module|Function|Class): [^\]]+\]\n?")


def _strip_embed_prefix(text: str) -> str:
    """Remove the ``[Source: ...]`` / ``[Module: ...]`` prefix before embedding.

    The prefix is useful for display but inflates cosine similarity — the
    generic structural tokens match loosely against any query.
    """
    return _EMBED_PREFIX_RE.sub("", text).strip()


def _chunk_text(text: str, source_file: str) -> list[ChunkMeta]:
    """Split docs into section-aware chunks.

    RST files are parsed with ``docutils`` for accurate section extraction.
    Markdown files use header-based splitting.  TOML files are split on
    ``[section]`` headers.  Navigation-only sections are skipped.  Long
    sections are split with character-based overlap.

    Each chunk is prefixed with ``[Source: <path> § <section>]``.
    """
    short_source = source_file
    if "divi/" in source_file:
        short_source = source_file.split("divi/", 1)[-1]

    if source_file.endswith(".rst"):
        return _chunk_rst(text, source_file, short_source)
    if source_file.endswith(".toml"):
        return _chunk_toml(text, source_file, short_source)
    return _chunk_markdown(text, source_file, short_source)


def _chunk_rst(text: str, source_file: str, short_source: str) -> list[ChunkMeta]:
    """Parse RST with docutils and extract sections as chunks."""
    parser = RstParser()
    settings = OptionParser(components=(RstParser,)).get_default_values()
    settings.report_level = 4  # SEVERE — suppress Sphinx-role warnings
    settings.halt_level = 5
    doc = new_document(source_file, settings)
    parser.parse(text, doc)

    chunks: list[ChunkMeta] = []
    lines = text.splitlines(keepends=True)

    for node in doc.traverse(section):
        title_node = node.next_node(title)
        section_title = title_node.astext() if title_node else None

        if section_title and section_title.lower().strip() in SKIP_SECTIONS:
            continue

        # Extract text, skipping child sections and system messages
        section_text = "\n".join(
            child.astext()
            for child in node.children
            if not isinstance(child, (section, system_message))
        ).strip()
        section_text = _DOCUTILS_NOISE_RE.sub("", section_text).strip()

        if not section_text or len(section_text) < MIN_CHUNK_LENGTH:
            continue

        start_line = node.line or 1
        end_line = min(start_line + section_text.count("\n") + 5, len(lines))

        if section_title:
            prefix = f"[Source: {short_source} § {section_title}]\n"
        else:
            prefix = f"[Source: {short_source}]\n"

        if len(section_text) <= CHUNK_SIZE:
            chunks.append(
                ChunkMeta(
                    text=prefix + section_text,
                    source_file=source_file,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        else:
            section_lines = section_text.splitlines(keepends=True)
            _split_long_section(section_lines, prefix, source_file, start_line, chunks)

    # Fallback: no sections found → chunk whole file
    if not chunks:
        prefix = f"[Source: {short_source}]\n"
        if len(text) <= CHUNK_SIZE:
            chunks.append(
                ChunkMeta(
                    text=prefix + text.strip(),
                    source_file=source_file,
                    start_line=1,
                    end_line=len(lines),
                )
            )
        else:
            _split_long_section(lines, prefix, source_file, 1, chunks)

    return chunks


def _chunk_by_headers(
    text: str,
    source_file: str,
    short_source: str,
    header_re: re.Pattern[str],
    title_group: int,
    *,
    skip_sections: set[str] | None = None,
) -> list[ChunkMeta]:
    """Split text into chunks by matching header lines with *header_re*.

    Parameters
    ----------
    header_re:
        Compiled regex whose match on a stripped line signals a new section.
    title_group:
        Which capture group in *header_re* contains the section title.
    skip_sections:
        Optional set of lowercased section titles to skip entirely.
    """
    lines = text.splitlines(keepends=True)
    chunks: list[ChunkMeta] = []

    sections: list[tuple[str | None, int, int]] = []
    current_title: str | None = None
    current_start = 0

    for i, line in enumerate(lines):
        m = header_re.match(line.strip())
        if m:
            if i > current_start:
                sections.append((current_title, current_start, i - 1))
            current_title = m.group(title_group).strip()
            current_start = i

    if current_start < len(lines):
        sections.append((current_title, current_start, len(lines) - 1))

    for section_title, start, end in sections:
        if (
            skip_sections
            and section_title
            and section_title.lower().strip() in skip_sections
        ):
            continue

        section_lines = lines[start : end + 1]
        section_text = "".join(section_lines).strip()
        if not section_text or len(section_text) < MIN_CHUNK_LENGTH:
            continue

        if section_title:
            prefix = f"[Source: {short_source} § {section_title}]\n"
        else:
            prefix = f"[Source: {short_source}]\n"

        if len(section_text) <= CHUNK_SIZE:
            chunks.append(
                ChunkMeta(
                    text=prefix + section_text,
                    source_file=source_file,
                    start_line=start + 1,
                    end_line=end + 1,
                )
            )
        else:
            _split_long_section(section_lines, prefix, source_file, start + 1, chunks)

    return chunks


def _chunk_markdown(text: str, source_file: str, short_source: str) -> list[ChunkMeta]:
    """Split Markdown by header lines."""
    return _chunk_by_headers(
        text,
        source_file,
        short_source,
        _MD_HEADER_RE,
        title_group=2,
        skip_sections=SKIP_SECTIONS,
    )


def _chunk_toml(text: str, source_file: str, short_source: str) -> list[ChunkMeta]:
    """Split a TOML file by ``[section]`` headers."""
    return _chunk_by_headers(
        text,
        source_file,
        short_source,
        _TOML_SECTION_RE,
        title_group=1,
    )


def _split_long_section(
    lines: list[str],
    prefix: str,
    source_file: str,
    base_line: int,
    chunks: list[ChunkMeta],
) -> None:
    """Split a long section into overlapping chunks."""
    buf: list[str] = []
    buf_chars = 0
    start_line = base_line

    for i, line in enumerate(lines):
        buf.append(line)
        buf_chars += len(line)

        if buf_chars >= CHUNK_SIZE:
            chunks.append(
                ChunkMeta(
                    text=prefix + "".join(buf),
                    source_file=source_file,
                    start_line=start_line,
                    end_line=base_line + i,
                )
            )
            # Overlap
            overlap_chars = 0
            overlap_start = len(buf)
            while overlap_start > 0 and overlap_chars < CHUNK_OVERLAP:
                overlap_start -= 1
                overlap_chars += len(buf[overlap_start])

            buf = buf[overlap_start:]
            buf_chars = sum(len(l) for l in buf)
            start_line = base_line + i - len(buf) + 1

    if buf:
        chunks.append(
            ChunkMeta(
                text=prefix + "".join(buf),
                source_file=source_file,
                start_line=start_line,
                end_line=base_line + len(lines) - 1,
            )
        )


# ---------------------------------------------------------------------------
# Chunking — Python (.py) via AST
# ---------------------------------------------------------------------------


def _extract_python_units(text: str, source_file: str) -> list[ChunkMeta]:
    """Parse a Python file and extract function/class units as chunks.

    Each unit contains the fully qualified name, signature, and docstring.
    This avoids indexing noisy raw code like import blocks, constant
    definitions, and license headers.

    Parameters
    ----------
    text:
        The full Python file content.
    source_file:
        Path string stored in the chunk metadata.

    Returns
    -------
    list[ChunkMeta]
        One chunk per documented function, class, or module docstring.
    """
    try:
        tree = ast.parse(text, filename=source_file)
    except SyntaxError:
        return []

    # Derive a module-style path: divi/pipeline/_core.py → divi.pipeline._core
    short_source = source_file
    if "divi/" in source_file:
        short_source = source_file.split("divi/", 1)[-1]
    module_path = short_source.replace("/", ".").removesuffix(".py")

    chunks: list[ChunkMeta] = []

    # Module-level docstring — enriched with defined symbols
    module_doc = ast.get_docstring(tree)
    if module_doc:
        # Collect top-level public names to make the chunk more specific
        symbols = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                symbols.append(node.name)
            elif isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef
            ) and not node.name.startswith("_"):
                symbols.append(node.name)

        chunk_text = f"[Module: {module_path}]\n{module_doc}"
        if symbols:
            chunk_text += f"\nDefines: {', '.join(symbols)}"

        chunks.append(
            ChunkMeta(
                text=chunk_text,
                source_file=source_file,
                start_line=1,
                end_line=tree.body[0].end_lineno or 1,
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _extract_class(node, module_path, source_file, chunks)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Only top-level functions (methods are handled inside _extract_class)
            if not _is_method(node, tree):
                _extract_function(node, module_path, source_file, chunks)

    # Capture `if __name__ == "__main__"` blocks — these contain runnable
    # examples (especially in tutorials) that are critical for code queries.
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            is_tutorial = "tutorial" in source_file.lower()
            if is_tutorial:
                # For tutorials, include the ENTIRE file (imports + main
                # block) so the model sees correct import paths.
                block_text = text.strip()
                start, end = 1, len(text.splitlines())
            else:
                start = node.lineno
                end = node.end_lineno or start
                block_text = "\n".join(text.splitlines()[max(0, start - 1) : end])

            if len(block_text) >= MIN_CHUNK_LENGTH:
                label = "Example" if is_tutorial else "Module"
                chunks.append(
                    ChunkMeta(
                        text=f"[{label}: {short_source}]\n{block_text}",
                        source_file=source_file,
                        start_line=start,
                        end_line=end,
                    )
                )

    return chunks


def _extract_class(
    node: ast.ClassDef,
    module_path: str,
    source_file: str,
    chunks: list[ChunkMeta],
) -> None:
    """Extract a class and its methods as chunks."""
    qualified = f"{module_path}.{node.name}"
    class_doc = ast.get_docstring(node)

    # Build class header: name + bases + docstring
    bases = [_name_of(b) for b in node.bases]
    bases_str = f"({', '.join(bases)})" if bases else ""
    header = f"[Class: {qualified}]\nclass {node.name}{bases_str}:"

    if class_doc:
        header += f'\n    """{class_doc}"""'

    chunks.append(
        ChunkMeta(
            text=header,
            source_file=source_file,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        )
    )

    # Extract documented methods
    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            _extract_function(child, qualified, source_file, chunks)


def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent_path: str,
    source_file: str,
    chunks: list[ChunkMeta],
) -> None:
    """Extract a function/method as a chunk (signature + docstring)."""
    doc = ast.get_docstring(node)
    if not doc:
        return  # Skip undocumented functions — not useful for RAG

    qualified = f"{parent_path}.{node.name}"
    sig = _get_signature(node)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"

    chunk_text = f"[Function: {qualified}]\n{prefix} {node.name}{sig}:\n"
    chunk_text += f'    """{doc}"""'

    chunks.append(
        ChunkMeta(
            text=chunk_text,
            source_file=source_file,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        )
    )


def _get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a function's parameter signature from the AST.

    Includes type annotations when present so that retrieved chunks
    show the expected types (e.g. ``method: ScipyMethod`` rather than
    just ``method``).
    """
    args = node.args
    parts: list[str] = []

    # positional args
    for arg in args.args:
        parts.append(_annotated_arg(arg))

    # *args
    if args.vararg:
        parts.append(f"*{_annotated_arg(args.vararg)}")

    # keyword-only args
    for arg in args.kwonlyargs:
        parts.append(_annotated_arg(arg))

    # **kwargs
    if args.kwarg:
        parts.append(f"**{_annotated_arg(args.kwarg)}")

    return f"({', '.join(parts)})"


def _annotated_arg(arg: ast.arg) -> str:
    """Return ``name: Type`` if the argument has an annotation, else just ``name``."""
    if arg.annotation is not None:
        return f"{arg.arg}: {_annotation_str(arg.annotation)}"
    return arg.arg


def _annotation_str(node: ast.expr) -> str:
    """Convert an annotation AST node to a readable string."""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_annotation_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_annotation_str(node.value)}[{_annotation_str(node.slice)}]"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f"{_annotation_str(node.left)} | {_annotation_str(node.right)}"
    if isinstance(node, ast.Tuple):
        return ", ".join(_annotation_str(e) for e in node.elts)
    if isinstance(node, ast.List):
        return "[" + ", ".join(_annotation_str(e) for e in node.elts) + "]"
    # Fallback: try ast.unparse (Python 3.9+), else dump
    try:
        return ast.unparse(node)
    except AttributeError:
        return ast.dump(node)


def _name_of(node: ast.expr) -> str:
    """Get a readable name from an AST expression (for base classes)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    return "..."


def _is_method(node: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.Module) -> bool:
    """Check whether *node* is a method inside a class (not top-level)."""
    for cls_node in ast.walk(tree):
        if isinstance(cls_node, ast.ClassDef) and node in cls_node.body:
            return True
    return False


# ---------------------------------------------------------------------------
# Chunking — Test files (.py under tests/)
# ---------------------------------------------------------------------------

MAX_TEST_CHUNK_CHARS = 1500


MAX_IMPORT_BLOCK_CHARS = 400


def _extract_import_block(lines: list[str]) -> str:
    """Extract the import block from the top of a Python file.

    Capped at :data:`MAX_IMPORT_BLOCK_CHARS` to avoid bloating chunks
    with test files that import dozens of symbols.
    """
    import_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith(("import ", "from "))
            or stripped.startswith("#")
            or stripped == ""
        ):
            import_lines.append(line)
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            # Skip module docstrings
            continue
        else:
            break
    result = "".join(import_lines).strip()
    if len(result) > MAX_IMPORT_BLOCK_CHARS:
        # Keep only the first N chars, ending at a line boundary
        truncated = result[:MAX_IMPORT_BLOCK_CHARS]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            result = truncated[:last_newline]
    return result


def _extract_test_usage(text: str, source_file: str) -> list[ChunkMeta]:
    """Extract test functions as usage-example chunks.

    Unlike :func:`_extract_python_units`, this extracts **full function
    bodies** because test functions demonstrate real API usage patterns.

    Parameters
    ----------
    text:
        The full Python file content.
    source_file:
        Path string stored in the chunk metadata.

    Returns
    -------
    list[ChunkMeta]
        One chunk per test function (skipping overly long ones).
    """
    try:
        tree = ast.parse(text, filename=source_file)
    except SyntaxError:
        return []

    lines = text.splitlines(keepends=True)
    file_imports = _extract_import_block(lines)

    short_source = source_file
    if "divi/" in source_file:
        short_source = source_file.split("divi/", 1)[-1]

    chunks: list[ChunkMeta] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue

        start = node.lineno
        end = node.end_lineno or start
        func_text = "".join(lines[max(0, start - 1) : end])

        if len(func_text) > MAX_TEST_CHUNK_CHARS:
            continue

        # Heuristic: skip heavily-mocked tests (not good usage examples)
        if func_text.count("mocker.patch") + func_text.count("mocker.spy") > 2:
            continue

        prefix = f"[Test: {short_source}::{node.name}]\n"
        chunk_text = prefix
        if file_imports:
            chunk_text += file_imports + "\n\n"
        chunk_text += func_text.strip()

        chunks.append(
            ChunkMeta(
                text=chunk_text,
                source_file=source_file,
                start_line=start,
                end_line=end,
                chunk_type="test",
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------


def _should_skip(path: Path) -> bool:
    """Return ``True`` if *path* should be excluded from indexing."""
    if path.name in SKIP_NAMES:
        return True
    is_test_file = "tests" in path.parts
    # Only apply SKIP_PREFIXES to non-test files (we want test_*.py)
    if not is_test_file and path.stem.startswith(SKIP_PREFIXES):
        return True
    # Only index files under allowed top-level directories
    if not any(part in INCLUDE_DIRS for part in path.parts):
        return True
    # Skip the AI module itself — it's the chatbot, not library documentation
    parts = path.parts
    if (
        "divi" in parts
        and "ai" in parts
        and parts.index("ai") == parts.index("divi") + 1
    ):
        return True
    # Still skip __pycache__ and build dirs within included dirs
    if any(part in {"__pycache__", "_build", ".git"} for part in path.parts):
        return True
    return False


def _collect_files(source_dirs: list[Path]) -> list[Path]:
    """Collect indexable files from *source_dirs*, respecting ``.gitignore``.

    Uses ``git ls-files`` so the skip list stays in sync with ``.gitignore``
    automatically.  Falls back to a simple recursive walk if ``git`` is
    unavailable or the directory is not inside a repository.
    """
    files: list[Path] = []
    for base in source_dirs:
        if base.is_file():
            if base.suffix in EXTENSIONS and not _should_skip(base):
                files.append(base)
            continue

        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=str(base),
                capture_output=True,
                text=True,
                check=True,
            )
            for rel in result.stdout.splitlines():
                path = base / rel
                if (
                    path.is_file()
                    and path.suffix in EXTENSIONS
                    and not _should_skip(path)
                ):
                    files.append(path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # git not available or not a repo — fall back to simple walk
            for path in sorted(base.rglob("*")):
                if (
                    path.is_file()
                    and path.suffix in EXTENSIONS
                    and not _should_skip(path)
                ):
                    files.append(path)

    # Always include pyproject.toml for project metadata (Python version, etc.)
    for base in source_dirs:
        pyproject = base / "pyproject.toml"
        if pyproject.is_file() and pyproject not in files:
            files.append(pyproject)

    return files


# ---------------------------------------------------------------------------
# Index building (CI / developer)
# ---------------------------------------------------------------------------


def build_index(
    source_dirs: list[Path],
    output_dir: Path,
    batch_size: int = 16,
) -> tuple[faiss.IndexFlatIP, list[ChunkMeta]]:
    """Build a FAISS index from source files and write it to *output_dir*.

    This is intended for CI or local development — the output files
    (``vectors.faiss`` and ``chunks.json``) are shipped in the
    ``divi/ai/_data/`` package directory.

    Parameters
    ----------
    source_dirs:
        Directories (and individual files) to scan.
    output_dir:
        Directory to write the index files into.

    Returns
    -------
    tuple[faiss.IndexFlatIP, list[ChunkMeta]]
        The FAISS inner-product index and the corresponding chunk metadata.
    """
    files = _collect_files(source_dirs)

    # Chunk all files — route by extension
    all_chunks: list[ChunkMeta] = []
    py_count = 0
    doc_count = 0
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue

        is_test = "tests" in fpath.parts
        is_tutorial = "tutorial" in str(fpath).lower()

        if fpath.suffix == ".py" and is_test:
            new_chunks = _extract_test_usage(text, str(fpath))
            py_count += 1
        elif fpath.suffix == ".py":
            new_chunks = _extract_python_units(text, str(fpath))
            # For short tutorials without an [Example:] chunk, add full file
            if is_tutorial and len(text) < 2500:
                has_example = any("[Example:" in c.text for c in new_chunks)
                if not has_example:
                    new_chunks.append(
                        ChunkMeta(
                            text=f"[Example: {str(fpath).split('divi/', 1)[-1]}]\n{text.strip()}",
                            source_file=str(fpath),
                            start_line=1,
                            end_line=len(text.splitlines()),
                            chunk_type="tutorial",
                        )
                    )
            py_count += 1
        else:
            new_chunks = _chunk_text(text, str(fpath))
            for c in new_chunks:
                c.chunk_type = "doc"
            doc_count += 1

        all_chunks.extend(new_chunks)

    # Synthetic chunk for install / Python version so retrieval can find it
    _install_chunk = ChunkMeta(
        text=(
            "Install: pip install divi. "
            "Python: see pyproject.toml [project].requires-python. "
            "Install and Python version requirements are in the project metadata."
        ),
        source_file="PROJECT:install",
        start_line=0,
        end_line=0,
    )
    all_chunks.append(_install_chunk)

    if not all_chunks:
        msg = "No documents found to index."
        raise RuntimeError(msg)

    # Embed with progress bar — leave 2 cores free so the system stays usable
    max_threads = max(1, (os.cpu_count() or 4) - 2)
    print(
        f"Embedding {len(all_chunks)} chunks "
        f"({py_count} .py files, {doc_count} doc files) "
        f"using {max_threads} threads …"
    )
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL, threads=max_threads)

    # Embed the body text only — strip [Source: ...] / [Module: ...] prefix
    # lines that inflate cosine similarity for every chunk.
    texts = [_strip_embed_prefix(c.text) for c in all_chunks]

    try:
        is_tty = Console().is_terminal
        if is_tty:
            vectors = list(
                track(
                    embedder.embed(texts, batch_size=batch_size),
                    total=len(texts),
                    description="Embedding …",
                )
            )
        else:
            vectors = []
            total = len(texts)
            for i, vec in enumerate(embedder.embed(texts, batch_size=batch_size), 1):
                vectors.append(vec)
                if i % 100 == 0 or i == total:
                    print(f"  Embedded {i}/{total} chunks", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — index not saved.")
        raise SystemExit(1)
    embeddings = np.array(vectors, dtype=np.float32)

    # Normalise for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "vectors.faiss"))
    (output_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in all_chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Index built: {len(all_chunks)} chunks → {output_dir}")

    return index, all_chunks


# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "_data"

# ---------------------------------------------------------------------------
# Index loading (end-user)
# ---------------------------------------------------------------------------

_DEV_INDEX_MISSING_MSG = (
    "[red bold]Search index not found.[/red bold]\n\n"
    "  Run [bold]python -m divi.ai build[/bold] to generate it.\n\n"
    "  [dim]Troubleshooting:[/dim]\n"
    "    • [bold]OOM:[/bold] python -m divi.ai build --batch-size 4\n"
    "    • [bold]Missing deps:[/bold] poetry install --with ai\n"
)

_PKG_INDEX_MISSING_MSG = (
    "[red bold]Search index not found.[/red bold]\n\n"
    "  The installed package is missing its search index.\n"
    "  Reinstall with [bold]pip install qoro-divi[ai][/bold]\n"
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _check_index_staleness() -> None:
    """Warn if the built index version doesn't match the project version."""
    meta_file = DATA_DIR / _META_FILE
    if not meta_file.exists() or not (_REPO_ROOT / ".git").exists():
        return
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        index_version = meta.get("project", {}).get("version")
        current_version = _read_pyproject(_REPO_ROOT).get("version")
        if index_version and current_version and index_version != current_version:
            Console().print(
                f"[yellow]Search index was built for v{index_version} "
                f"but the project is v{current_version}.[/yellow]\n"
                "  Run [bold]python -m divi.ai build[/bold] to update it.\n"
            )
    except (json.JSONDecodeError, OSError):
        pass


def load_index(index_dir: Path) -> tuple[faiss.IndexFlatIP, list[ChunkMeta]]:
    """Load a pre-built FAISS index and chunk metadata from *index_dir*.

    Parameters
    ----------
    index_dir:
        Directory containing ``vectors.faiss`` and ``chunks.json``,
        typically ``divi/ai/_data/``.

    Returns
    -------
    tuple[faiss.IndexFlatIP, list[ChunkMeta]]
        The FAISS index and corresponding chunk metadata.
    """
    index = faiss.read_index(str(index_dir / "vectors.faiss"))
    raw = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [ChunkMeta(**entry) for entry in raw]
    return index, chunks


def load_search_stack() -> tuple[faiss.IndexFlatIP, list[ChunkMeta], "TextEmbedding"]:
    """Load the full retrieval stack from the bundled index, or exit on failure.

    Returns
    -------
    tuple
        ``(index, chunks, embedder)`` ready for
        :func:`_retriever.retrieve`.

    Raises
    ------
    SystemExit
        If the index files are missing from :data:`DATA_DIR`.
    """
    if not (DATA_DIR / "vectors.faiss").exists():
        msg = (
            _DEV_INDEX_MISSING_MSG
            if (_REPO_ROOT / ".git").exists()
            else _PKG_INDEX_MISSING_MSG
        )
        Console().print(msg)
        raise SystemExit(1)

    _check_index_staleness()
    index, chunks = load_index(DATA_DIR)
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return index, chunks, embedder


# ---------------------------------------------------------------------------
# Project metadata (import map + project info)
# ---------------------------------------------------------------------------

_META_FILE = "project_meta.json"


def _scan_init_imports(package_dir: Path) -> dict[str, list[str]]:
    """Scan ``__init__.py`` files for public re-exports.

    Returns a mapping from ``divi.subpackage`` module paths to lists of
    exported names, e.g. ``{"divi.qprog": ["VQE", "QAOA", ...]}``.
    """
    imports: dict[str, list[str]] = {}

    for init_file in sorted(package_dir.rglob("__init__.py")):
        # Derive module path: divi/qprog/__init__.py -> divi.qprog
        rel = init_file.relative_to(package_dir.parent)
        module = str(rel.parent).replace("/", ".")

        try:
            tree = ast.parse(init_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        names: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    # Skip private names and star imports
                    if name.startswith("_") or name == "*":
                        continue
                    names.append(name)

        if names:
            imports[module] = names

    return imports


def _read_pyproject(repo_root: Path) -> dict[str, str]:
    """Extract key metadata from ``pyproject.toml``."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return {}

    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    meta: dict[str, str] = {}
    if "name" in project:
        meta["name"] = project["name"]
    if "version" in project:
        meta["version"] = project["version"]
    if "requires-python" in project:
        meta["python"] = project["requires-python"]
    return meta


def build_project_meta(repo_root: Path, output_dir: Path) -> dict:
    """Build and save project metadata (import map + project info).

    Scans ``__init__.py`` files for public exports and reads
    ``pyproject.toml`` for version and Python requirements.
    Saves the result as ``project_meta.json`` in *output_dir*.
    """
    package_dir = repo_root / "divi"
    import_map = _scan_init_imports(package_dir)
    project_info = _read_pyproject(repo_root)

    # Modules to skip entirely
    _SKIP = {".ai", ".reporting", ".pipeline", ".stages"}

    # Build deduplicated import lines — prefer the shortest module path
    # for each name (e.g. VQE from divi.qprog, not divi.qprog.algorithms)
    name_to_module: dict[str, str] = {}
    for module, names in sorted(import_map.items()):
        # Skip root 'divi' (only has enable_logging)
        if module == "divi":
            continue
        # Skip private sub-packages (e.g. divi.circuits._cirq)
        if "._" in module:
            continue
        # Skip internal modules
        if any(skip in module for skip in _SKIP):
            continue

        for name in names:
            existing = name_to_module.get(name)
            if existing is None or len(module) < len(existing):
                name_to_module[name] = module

    # Group names by module for readable output
    module_to_names: dict[str, list[str]] = {}
    for name, module in sorted(name_to_module.items()):
        module_to_names.setdefault(module, []).append(name)

    import_lines = [
        f"from {mod} import {', '.join(sorted(names))}"
        for mod, names in sorted(module_to_names.items())
    ]

    meta = {
        "import_lines": import_lines,
        "project": project_info,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Project metadata saved: {len(import_lines)} import lines")

    return meta


def load_project_meta() -> dict | None:
    """Load project metadata from the bundled data directory.

    Returns ``None`` if the metadata file does not exist.
    """
    path = DATA_DIR / _META_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
