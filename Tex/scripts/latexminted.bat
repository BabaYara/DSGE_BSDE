@echo off
setlocal
set "PYTHONPATH=%~dp0latexminted_lib;%PYTHONPATH%"
python -m latexminted.cmdline %*
