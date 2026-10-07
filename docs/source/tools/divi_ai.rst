divi-ai: AI Coding Assistant
============================

.. important::

   divi-ai is **experimental**. It runs on CPU with small local models, so
   answers may be inaccurate and knowledge is limited to what was indexed at
   build time. Always verify code against the
   `official documentation <https://divi.readthedocs.io/en/latest/>`_.

**divi-ai** is a coding assistant for Divi that runs directly in your terminal.
It answers questions, generates code examples, and explains APIs — all using a
local LLM on your machine. No API keys are required. An internet connection is
needed initially to cache the generation and retrieval models. Once those
models are cached, divi-ai works offline.

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
   :widths: 20 55 25

   * - Key
     - Model
     - Context
   * - ``4b`` (default)
     - Qwen 3.5 4B Q4_K_M
     - 8K
   * - ``9b``
     - Qwen 3.5 9B Q4_K_M
     - 8K

The 4B model is the default because it offers the best balance of grounded
Divi answers and CPU generation speed. The 9B model is more consistent on
multi-step API questions, but takes substantially longer to answer.

**Hardware recommendations:**

* 16+ GB RAM: ``4b`` for speed or ``9b`` for higher answer quality
* Less than 16 GB RAM: ``4b``

First Launch
------------

.. code-block:: bash

   divi-ai

On the first run:

1. The interactive model selector opens (arrow keys to navigate, Enter to
   confirm).
2. The selected generation model and the retrieval embedding model are cached.
   The first question may also download the relevance model. Keep the machine
   online until the first question completes successfully.
3. The model and search index are loaded into memory. This can take
   30--60 seconds depending on your hardware.
4. The TUI opens and you can start asking questions.

Subsequent launches reuse the cached models. The cache location is
platform-dependent and determined by ``platformdirs``. If an upgrade removes
the selected model from the catalogue, divi-ai asks you to choose again. It
does not delete the old model files automatically; obsolete model folders can
be removed from the cache to recover disk space.

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
   Number of documentation chunks retrieved per query (default: 3).
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
   On Windows this is the most common installation failure.
   ``llama-cpp-python`` has no prebuilt wheels on PyPI, so
   ``pip install`` always downloads the source and compiles it. On
   Linux and macOS this usually succeeds silently when a C++ toolchain
   is present; on Windows it frequently fails. Try these in order:

   1. **Use a prebuilt CPU wheel from abetlen's index**, when one is available
      for your platform:

      .. code-block:: bash

         pip install "qoro-divi[ai]" \
             --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
             --only-binary=llama-cpp-python

      Divi's dependency constraint selects a compatible version.
      ``--only-binary=llama-cpp-python`` makes pip report an error if the wheel
      index has no compatible build instead of silently compiling that package
      from source.

   2. **If you must build from source**, install a C++ toolchain:

      * **Linux:** ``build-essential`` (Debian/Ubuntu) or equivalent.
      * **macOS:** Xcode Command Line Tools (``xcode-select --install``).
      * **Windows:** Visual Studio Build Tools with the *Desktop
        development with C++* workload. Run ``pip install`` from an
        *x64 Native Tools Command Prompt for VS* so MSVC's ``cl.exe``
        is on ``PATH``.

   .. note::

      **Windows: Strawberry Perl on PATH.** If a Windows source build
      fails and the build log shows ``C:/Strawberry/c/bin/gcc.exe`` or
      any path under ``C:\Strawberry``, CMake is picking up the MinGW
      compiler bundled with Strawberry Perl instead of MSVC. The
      vendored ``llama.cpp`` sources do not build cleanly with that
      toolchain. Either remove ``C:\Strawberry\c\bin`` from ``PATH``
      in the current Command Prompt window, or launch an *x64 Native
      Tools Command Prompt for VS* before running pip.

**"Context window exceeded" / answers cut off mid-sentence**
   The conversation has filled the model's context window. Use ``/reset``
   to clear history.

**Slow or unusable on my machine**
   Use the default ``4b`` model. If it is still too slow, divi-ai may not be
   practical on your system.

**Answers seem wrong or hallucinated**
   Try a larger model if your hardware allows it. See the important
   notice at the top of this page.

.. _divi-ai-dev-tools:

For Contributors
----------------

These commands are for Divi contributors rebuilding or evaluating the
search index. Install the AI dependencies first with ``uv sync --extra ai``
to pull in only the AI stack, or ``uv sync --all-extras`` if you are also
working on docs, tests, or other areas that need the full development
environment.

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
