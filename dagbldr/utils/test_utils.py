import os
from nose import SkipTest, with_setup


def check_skip_travis():
    """Skip test if being run on Travis."""
    if os.environ.get('TRAVIS') == "true":
        raise SkipTest("This test needs to be skipped on Travis")

with_travis = with_setup(check_skip_travis)
