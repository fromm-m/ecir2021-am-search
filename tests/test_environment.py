import os
import pytest

from arclus.settings import PREP_ASSIGNMENTS, PREP_CLAIMS, PREP_PREMISES


@pytest.mark.environment_test
def test_environment():
    assert (os.path.exists(PREP_PREMISES))
    assert (os.path.exists(PREP_ASSIGNMENTS))
    assert (os.path.exists(PREP_CLAIMS))
