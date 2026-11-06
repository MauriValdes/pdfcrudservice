
@echo off
cd /d "%~dp0"

call .venv\Scripts\activate.bat

python -m pytest tests/test_docs_api.py tests/test_document_service.py -v

pause
