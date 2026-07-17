"""
conftest.py — sets VECTORDB_DATA_DIR to a temp directory before any test
module imports main.py. main.py loads/saves persisted state at import
time (as a module-level side effect), so this has to happen before the
first `import main` anywhere in the test session, not inside a fixture.
"""

import os
import sys
import tempfile
from pathlib import Path

# Make the project root (parent of tests/) importable regardless of
# where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent))

_tmp_data_dir = tempfile.mkdtemp(prefix="vectordb_test_data_")
os.environ["VECTORDB_DATA_DIR"] = _tmp_data_dir

# If sentence-transformers happens to be installed in the test environment
# but the model isn't cached locally, huggingface_hub will otherwise retry
# network requests for a long time before failing. Force fast-fail instead
# so tests stay fast and deterministic regardless of what's installed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# In this sandbox, importing sentence_transformers (torch) hangs for a
# very long time regardless of network settings — likely slow storage
# for the multi-GB CUDA wheel files, unrelated to the app's own logic.
# Tests must never trigger a real sentence-transformers load; anything
# that needs to test that code path does so with mocks (see
# tests/test_ai_providers.py). This patches the loader to fail instantly
# instead of attempting a real import.
import ai_providers  # noqa: E402


def _never_actually_load(self):
    self._load_failed = True


ai_providers.SentenceTransformerEmbedder._load = _never_actually_load
