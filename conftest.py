"""Make the project root importable for tests (so `import main` works)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
