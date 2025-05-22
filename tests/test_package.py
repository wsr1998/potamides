"""Test the package itself."""

import importlib.metadata

import potamides as m


def test_version():
    assert importlib.metadata.version("potamides") == m.__version__
