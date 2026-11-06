import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_document_service.py",
        "tests/test_docs_api.py",
        "-v",
    ],
    cwd=PROJECT_ROOT,
    capture_output=True,
    text=True,
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
