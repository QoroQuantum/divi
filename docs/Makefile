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
	@echo "  html       - Build HTML documentation"
	@echo "  clean      - Remove all build files"
	@echo "  live       - Build HTML and serve locally (auto-reload on changes)"
	@echo "  serve      - Serve built HTML documentation locally"
	@echo "  linkcheck  - Check for broken links"
	@echo "  doctest    - Run doctests in documentation"
	@echo "  coverage   - Run coverage check"
	@echo "  spelling   - Run spell checker"
	@echo "  check      - Run all checks (linkcheck, doctest, coverage)"
	@echo "  watch      - Watch for changes and rebuild automatically"
	@echo "  install    - Install documentation dependencies"
	@echo ""
	@echo "Variables:"
	@echo "  PORT       - Port for serving documentation (default: 8000)"
	@echo "  SPHINXOPTS - Additional Sphinx options"
	@echo ""
	@echo "Examples:"
	@echo "  make html PORT=8080    - Build HTML and serve on port 8080"
	@echo "  make live              - Build and serve with auto-reload"
	@echo "  make check             - Run all quality checks"

.PHONY: help Makefile

# Build targets
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "HTML documentation built in $(BUILDDIR)/html/"


# Clean target
clean:
	@echo "Cleaning build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Build directory cleaned."

# Serve documentation locally
serve: html
	@echo "Serving documentation at http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop the server"
	@cd "$(BUILDDIR)/html" && python -m http.server $(PORT)

# Live reload (requires sphinx-autobuild)
live:
	@echo "Starting live documentation server..."
	@echo "Documentation will be available at http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop the server"
	@if [ -d "$(BUILDDIR)/html" ]; then \
		echo "Using existing HTML build for faster startup..."; \
		sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" --port $(PORT) --host 0.0.0.0 --open-browser --re-ignore ".*\.(pyc|pyo)$$"; \
	else \
		echo "No existing build found, building from scratch..."; \
		sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" --port $(PORT) --host 0.0.0.0 --open-browser; \
	fi

# Watch for changes and rebuild
watch:
	@echo "Watching for changes in $(SOURCEDIR)..."
	@echo "Press Ctrl+C to stop watching"
	@while true; do \
		inotifywait -r -e modify,create,delete "$(SOURCEDIR)"; \
		echo "Changes detected, rebuilding..."; \
		$(MAKE) html; \
		echo "Rebuild complete. Watching for more changes..."; \
	done

# Quality checks
linkcheck:
	@echo "Checking for broken links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo "Link check complete. See $(BUILDDIR)/linkcheck/output.txt for results."

doctest:
	@echo "Running doctests..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo "Doctests complete. See $(BUILDDIR)/doctest/output.txt for results."

coverage:
	@echo "Running coverage check..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Coverage check complete. See $(BUILDDIR)/coverage/python.txt for results."

spelling:
	@echo "Running spell checker..."
	@$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS) $(O)
	@echo "Spell check complete. See $(BUILDDIR)/spelling/output.txt for results."

# Run all checks
check: linkcheck doctest coverage
	@echo "All checks completed!"

# Install dependencies
install:
	@echo "Installing documentation dependencies..."
	@poetry install --with docs
	@echo "Dependencies installed!"

# Development helpers
open:
	@echo "Opening documentation in browser..."
	@xdg-open "$(BUILDDIR)/html/index.html" 2>/dev/null || \
	 open "$(BUILDDIR)/html/index.html" 2>/dev/null || \
	 echo "Please open $(BUILDDIR)/html/index.html in your browser"

# Quick development cycle
dev: clean html serve

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
