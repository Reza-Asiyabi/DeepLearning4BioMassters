"""Root pytest configuration.

Adds ``src/`` to ``sys.path`` so that ``import biomassters`` works in every
test file without requiring a fully working editable install.  This is the
standard approach for src-layout projects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
