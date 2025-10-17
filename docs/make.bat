@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

REM Default build directory
if "%BUILDDIR%" == "" set BUILDDIR=_build

REM Default sphinx-build command
if "%SPHINXBUILD%" == "" set SPHINXBUILD=sphinx-build

if "%1" == "" goto help

%SPHINXBUILD% -M %1 . %BUILDDIR% %SPHINXOPTS% %O%
if errorlevel 1 goto error
goto end

:help
echo.Please use `make <target>` where <target> is one of the following:
echo.  html       to make standalone HTML files
echo.  dirhtml    to make a directory of HTML files
echo.  singlehtml to make a single large HTML file
echo.  latex      to make LaTeX files, you can set LATEXOPTS environment variable
echo.  man        to make manual pages
echo.  text       to make text files
echo.  epub       to make an epub document
echo.  gettext    to make PO message files
echo.  json       to make JSON files
echo.  doctest    to run doctests
echo.  linkcheck  to check all external links for integrity
echo.  xml        to make XML files
echo.  clean      to remove generated files

:error
echo.
echo.Build failed.
goto end

:end
popd
EXIT /B %errorlevel%