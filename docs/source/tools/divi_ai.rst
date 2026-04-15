divi-ai: AI Coding Assistant
============================

.. important::

   divi-ai is **experimental**. It runs on CPU with small local models, so
   answers may be inaccurate and knowledge is limited to what was indexed at
   build time. Always verify code against the
   `official documentation <https://divi.readthedocs.io/en/latest/>`_.

**divi-ai** is a coding assistant for Divi that runs directly in your terminal.
It answers questions, generates code examples, and explains APIs — all using a
local LLM on your machine. No API keys required. After the first launch
(which downloads the model), divi-ai works fully offline.

Installation
------------

.. code-block:: bash

   pip install qoro-divi[ai]

If installation fails due to ``llama-cpp-python``, see
:ref:`Troubleshooting <divi-ai-troubleshooting>` below.

.. _choosing-a-model:

Choosing a Model
----------------

On first launch, an interactive selector lets you pick a model. Choose one
before launching so you know what to expect:

.. list-table::
   :header-rows: 1
   :widths: 14 25 10 12 12

   * - Key
     - Model
     - Download
     - Est. RAM
     - Context
   * - ``1.5b``
     - Qwen 2.5 Coder 1.5B
     - 1.0 GB
     - ~1.2 GB
     - 8K
   * - ``3b``
     - Qwen 2.5 Coder 3B
     - 1.9 GB
     - ~2.3 GB
     - 8K
   * - ``7b`` (default)
     - Qwen 2.5 Coder 7B
     - 4.5 GB
     - ~5.4 GB
     - 16K
   * - ``14b``
     - Qwen 2.5 Coder 14B
     - 8.4 GB
     - ~10.1 GB
     - 16K
   * - ``e2b``
     - Gemma 4 E2B
     - 2.9 GB
     - ~3.5 GB
     - 8K
   * - ``e4b``
     - Gemma 4 E4B
     - 4.6 GB
     - ~5.5 GB
     - 8K

The **Qwen Coder** models are code-specialized and generally give better
results for code generation. The **Gemma** models are general-purpose
alternatives that work well for explanations and conceptual questions.
Larger models produce better answers but need more RAM and run slower.

**Hardware recommendations:**

* Apple Silicon with 16+ GB RAM: ``7b`` or ``14b``
* x86 with 32+ GB RAM: ``7b`` or ``14b``
* x86 with 16+ GB RAM: ``e4b`` or ``7b``
* Less than 16 GB RAM: ``1.5b``, ``3b``, or ``e2b``

First Launch
------------

.. code-block:: bash

   divi-ai

On the first run:

1. The interactive model selector opens (arrow keys to navigate, Enter to
   confirm).
2. The selected model is downloaded from HuggingFace (~1--9 GB depending on
   your choice). This requires an internet connection and may take a few
   minutes.
3. The model and search index are loaded into memory. This can take
   30--60 seconds depending on your hardware.
4. The TUI opens and you can start asking questions.

Subsequent launches skip the download step and are much faster. Models are
cached locally (the exact location is platform-dependent, determined by
``platformdirs``). Delete model folders from the cache directory to free
disk space.

Using the Chat Interface
------------------------

Type a question and press Enter to get an answer. The header bar tracks
how much of the model's context window your conversation has used.

* Press **Escape** to cancel generation mid-stream.
* Use ``/reset`` before switching to a new topic to free context.
* Use ``/retry`` if the answer seems incomplete or off.

Slash Commands
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``/save <file>``
     - Save the last code block to a file (relative to your working directory).
       Automatically runs a syntax check on the saved file.
   * - ``/copy``
     - Copy the last code block to the clipboard.
       Requires ``xclip`` or ``xsel`` on Linux.
   * - ``/check``
     - Syntax-check all Python code blocks from the last response.
   * - ``/retry``
     - Re-run the query (including retrieval) to get a different response.
   * - ``/reset``
     - Clear conversation history and free context window space.
   * - ``/clear``
     - Clear the screen and reset history.
   * - ``/quit``, ``/exit``
     - Exit the TUI.

CLI Options
-----------

.. code-block:: text

   divi-ai [OPTIONS]

``--reselect-model``
   Forget the saved model preference and re-prompt for selection.

``--top-k N``
   Number of documentation chunks retrieved per query (default: 8).
   Higher values give the model more context but use more of the context
   window. Lower values are faster but may miss relevant information.

``--max-tokens N``
   Maximum tokens the model can generate per response (default: 1024).

``--debug``
   Show index loading info and library messages.

``--dev``
   Developer mode: show retrieved chunks, FAISS scores, sources, and
   token generation speed after each response.

.. _divi-ai-troubleshooting:

Troubleshooting
---------------

**llama-cpp-python fails to install**
   This is the most common issue. The package compiles C++ at install time
   and needs a working toolchain: ``build-essential`` on Debian/Ubuntu,
   Xcode Command Line Tools on macOS, or Visual Studio Build Tools on
   Windows. On some systems, installing a prebuilt wheel helps:
   ``pip install llama-cpp-python --prefer-binary``

**"Context window exceeded" / answers cut off mid-sentence**
   The conversation has filled the model's context window. Use ``/reset``
   to clear history. If this happens frequently, switch to a model with
   a 16K context window (``7b`` or ``14b``).

**Slow or unusable on my machine**
   Try ``1.5b`` or ``e2b`` — they run acceptably on most hardware. If
   even those are too slow, divi-ai may not be practical on your system.

**Answers seem wrong or hallucinated**
   Try a larger model if your hardware allows it. See the important
   notice at the top of this page.

.. _divi-ai-dev-tools:

For Contributors
----------------

These commands are for Divi contributors rebuilding or evaluating the
search index. Install the AI dependencies first: ``uv sync --extra ai``

.. code-block:: bash

   python -m divi.ai help       # Show commands and workflow overview
   python -m divi.ai build      # Rebuild the FAISS index from source
   python -m divi.ai search     # Interactive search against the index
   python -m divi.ai inspect    # Inspect assembled prompts (no LLM)
   python -m divi.ai eval       # Run eval queries, save results
   python -m divi.ai compare    # Compare two eval runs side-by-side

Typical development workflow:

1. Change source code or docs.
2. ``python -m divi.ai build`` to rebuild the index.
3. ``python -m divi.ai search`` or ``inspect`` to verify retrieval quality.
4. ``divi-ai`` to test end-to-end.

.. note::

   If you run out of memory during ``build``, reduce the batch size:
   ``python -m divi.ai build --batch-size 4``
