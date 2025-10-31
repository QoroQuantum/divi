# Enhanced Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PORT          ?= 8000

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Development:"
	@echo "    dev        - Start a live-reloading development server"
	@echo "    serve      - Build and serve the documentation for preview"
	@echo "    build      - Build the HTML documentation"
	@echo "    clean      - Remove all build files"
	@echo ""
	@echo "  Quality Checks:"
	@echo "    test       - Run all quality checks (linkcheck, spelling, coverage)"
	@echo "    linkcheck  - Check for broken links"
	@echo "    spelling   - Check for spelling errors"
	@echo "    coverage   - Run a documentation coverage check"
	@echo ""
	@echo "  Deployment & Other:"
	@echo "    install    - Install documentation dependencies"
	@echo "    open       - Open the built documentation in your browser"
	@echo ""
	@echo "Variables:"
	@echo "  PORT       - Port for serving documentation (default: 8000)"
	@echo "  SPHINXOPTS - Additional Sphinx options"
	@echo ""
	@echo "Examples:"
	@echo "  make dev PORT=8080       - Start the dev server on port 8080"
	@echo "  make test                - Run all quality checks"

.PHONY: help Makefile build clean serve dev open linkcheck coverage spelling test install

KNOWN_TARGETS := help dev serve build clean test linkcheck spelling coverage install open

# ==============================================================================
# Development
# ==============================================================================

build:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) || exit 1
	@if [ ! -d "$(BUILDDIR)/html" ]; then \
		echo "Error: Build failed - directory $(BUILDDIR)/html was not created."; \
		exit 1; \
	fi
	@echo "HTML documentation built in $(BUILDDIR)/html/"

clean:
	@echo "Cleaning build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Build directory cleaned."

serve: build
	@echo "Serving documentation at http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop the server"
	@if [ ! -d "$(BUILDDIR)/html" ]; then \
		echo "Error: Build directory $(BUILDDIR)/html does not exist. Run 'make build' first."; \
		exit 1; \
	fi
	@cd "$(BUILDDIR)/html" && python -m http.server $(PORT)

dev:
	@echo "Starting live documentation server..."
	@echo "Documentation will be available at http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop the server"
	@if [ -d "$(BUILDDIR)/html" ]; then \
		echo "Using existing HTML build for faster startup..."; \
		sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" --port $(PORT) --host 0.0.0.0 --open-browser --re-ignore ".*\.(pyc|pyo)$$"; \
	else \
		echo "No existing build found, building from scratch..."; \
		sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" --port $(PORT) --host 0.0.0.0 --open-browser --re-ignore ".*\.(pyc|pyo)$$"; \
	fi

open:
	@echo "Opening documentation in browser..."
	@xdg-open "$(BUILDDIR)/html/index.html" 2>/dev/null || \
	 open "$(BUILDDIR)/html/index.html" 2>/dev/null || \
	 echo "Please open $(BUILDDIR)/html/index.html in your browser"

# ==============================================================================
# Quality Checks
# ==============================================================================

linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXBUILD) -E -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check complete. See $(BUILDDIR)/linkcheck/output.txt for results."

coverage:
	@echo "Running coverage check..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Coverage check complete. See $(BUILDDIR)/coverage/python.txt for results."

spelling:
	@echo "Running spell checker..."
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS) $(O)
	@echo "Spell check complete. See $(BUILDDIR)/spelling/output.txt for results."

test: spelling linkcheck coverage
	@echo "All quality checks completed!"

# ==============================================================================
# Other
# ==============================================================================

install:
	@echo "Installing documentation dependencies..."
	@poetry install --with docs
	@echo "Dependencies installed!"

# Catch-all for other Sphinx commands
.DEFAULT:
	@TARGET='$@'; \
	KNOWN_TARGETS='$(KNOWN_TARGETS)'; \
	BEST_MATCH=''; \
	MIN_DIST=999; \
	for t in $$KNOWN_TARGETS; do \
		DIST=$$(echo "$$TARGET $$t" | awk ' \
			function min(x, y) { return x < y ? x : y } \
			function levenshtein(s1, s2, l1, l2, d, i, j, c) { \
				l1 = length(s1); l2 = length(s2); \
				for (i = 0; i <= l1; i++) d[i, 0] = i; \
				for (j = 0; j <= l2; j++) d[0, j] = j; \
				for (i = 1; i <= l1; i++) { \
					for (j = 1; j <= l2; j++) { \
						c = (substr(s1, i, 1) == substr(s2, j, 1)) ? 0 : 1; \
						d[i, j] = min(d[i-1, j] + 1, min(d[i, j-1] + 1, d[i-1, j-1] + c)); \
					} \
				} \
				return d[l1, l2]; \
			} \
			{ print levenshtein($$1, $$2) }'); \
		if [ $$DIST -lt $$MIN_DIST ]; then \
			MIN_DIST=$$DIST; \
			BEST_MATCH=$$t; \
		fi \
	done; \
	if [ $$MIN_DIST -le 3 ] && [ "$$BEST_MATCH" != "$$TARGET" ]; then \
		echo "make: '$$TARGET' is not a valid target. Did you mean '$$BEST_MATCH'?"; \
		exit 1; \
	else \
		@$(SPHINXBUILD) -M $$TARGET "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O); \
	fi
