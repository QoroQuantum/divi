@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
  set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build
if "%PORT%" == "" (
  set PORT=8000
)

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
  echo.
  echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
  echo.installed, then set the SPHINXBUILD environment variable to point
  echo.to the full path of the 'sphinx-build' executable. Alternatively you
  echo.may add the Sphinx directory to PATH.
  echo.
  echo.If you don't have Sphinx installed, grab it from
  echo.https://www.sphinx-doc.org/
  exit /b 1
)

if "%1" == "" goto help
if "%1" == "help" goto help

REM Custom targets
if "%1" == "build" goto build
if "%1" == "clean" goto clean
if "%1" == "serve" goto serve
if "%1" == "dev" goto dev
if "%1" == "spelling" goto spelling
if "%1" == "linkcheck" goto linkcheck
if "%1" == "coverage" goto coverage
if "%1" == "test" goto test
if "%1" == "install" goto install
if "%1" == "open" goto open

REM Fallback to Sphinx's built-in make mode for other targets
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:build
@echo Building HTML documentation...
%SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%/html" %SPHINXOPTS% %O%
if errorlevel 1 exit /b 1
@echo HTML documentation built in %BUILDDIR%/html/
goto end

:clean
@echo Cleaning build directory...
if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
@echo Build directory cleaned.
goto end

:serve
call :build
if errorlevel 1 exit /b 1
@echo Serving documentation at http://localhost:%PORT%
@echo Press Ctrl+C to stop the server
cd "%BUILDDIR%/html" && python -m http.server %PORT%
goto end

:dev
@echo Starting live documentation server...
@echo Documentation will be available at http://localhost:%PORT%
@echo Press Ctrl+C to stop the server
sphinx-autobuild -b html "%SOURCEDIR%" "%BUILDDIR%/html" --port %PORT% --host 0.0.0.0 --open-browser --re-ignore ".*\.(pyc|pyo)$"
goto end

:spelling
@echo Running spell checker...
%SPHINXBUILD% -b spelling "%SOURCEDIR%" "%BUILDDIR%/spelling" %SPHINXOPTS% %O%
@echo Spell check complete. See %BUILDDIR%/spelling/output.txt for results.
goto end

:linkcheck
@echo Checking for broken links...
%SPHINXBUILD% -E -b linkcheck "%SOURCEDIR%" "%BUILDDIR%/linkcheck" %SPHINXOPTS% %O%
@echo Link check complete. See %BUILDDIR%/linkcheck/output.txt for results.
goto end

:coverage
@echo Running coverage check...
%SPHINXBUILD% -b coverage "%SOURCEDIR%" "%BUILDDIR%/coverage" %SPHINXOPTS% %O%
@echo Coverage check complete. See %BUILDDIR%/coverage/python.txt for results.
goto end

:test
call :spelling
call :coverage
call :linkcheck
@echo All quality checks completed!
goto end

:install
@echo Installing documentation dependencies...
poetry install --with docs
@echo Dependencies installed!
goto end

:open
@echo Opening documentation in browser...
start "" "%BUILDDIR%/html/index.html"
goto end

:help
@echo Available targets:
@echo.
@echo   Development:
@echo     dev        - Start a live-reloading development server
@echo     serve      - Build and serve the documentation for preview
@echo     build      - Build the HTML documentation
@echo     clean      - Remove all build files
@echo.
@echo   Quality Checks:
@echo     test       - Run all quality checks (linkcheck, spelling, coverage)
@echo     linkcheck  - Check for broken links
@echo     spelling   - Check for spelling errors
@echo     coverage   - Run a documentation coverage check
@echo.
@echo   Deployment ^& Other:
@echo     install    - Install documentation dependencies
@echo     open       - Open the built documentation in your browser
@echo.
@echo Variables:
@echo   PORT       - Port for serving documentation (default: 8000)
@echo   SPHINXOPTS - Additional Sphinx options
@echo.
@echo Examples:
@echo   make.bat dev PORT=8080       - Start the dev server on port 8080
@echo   make.bat test                - Run all quality checks

:end
popd
