import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
# analyses created by the API tests go to a throwaway folder, never the app's data/ directory
os.environ["PITCHLAB_DATA_DIR"] = tempfile.mkdtemp(prefix="pitchlab-test-")

from pitchlab.config import DRIVELINE_DIR, MODELS_DIR, REFERENCE_DIR  # noqa: E402
from pitchlab.swing.config import DRIVELINE_DIR as SWING_DRIVELINE_DIR, REFERENCE_DIR as SWING_REFERENCE_DIR  # noqa: E402

needs_driveline = pytest.mark.skipif(not (DRIVELINE_DIR / "landmarks.csv").exists(),
                                     reason="Driveline data not downloaded (scripts/fetch_assets.py)")
needs_reference = pytest.mark.skipif(not (REFERENCE_DIR / "reference.json").exists(),
                                     reason="reference bundle not built (scripts/build_reference.py)")
needs_swing_driveline = pytest.mark.skipif(not (SWING_DRIVELINE_DIR / "landmarks.csv").exists(),
                                           reason="Driveline hitting data not downloaded (scripts/swing_fetch_assets.py)")
needs_swing_reference = pytest.mark.skipif(not (SWING_REFERENCE_DIR / "reference.json").exists(),
                                           reason="hitting reference not built (scripts/swing_build_reference.py)")
needs_pose_model = pytest.mark.skipif(not (MODELS_DIR / "pose_landmarker_lite.task").exists(),
                                      reason="pose models not downloaded (scripts/fetch_assets.py)")


@pytest.fixture(scope="session")
def trials():
    from pitchlab.driveline import load_trials
    return load_trials()


@pytest.fixture(scope="session")
def reference():
    from pitchlab.reference import Reference
    return Reference.load()


@pytest.fixture(scope="session")
def swing_trials():
    from pitchlab.swing.driveline import load_trials
    return load_trials()


@pytest.fixture(scope="session")
def swing_reference():
    from pitchlab.swing.reference import Reference
    return Reference.load()
