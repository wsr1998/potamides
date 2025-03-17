"""Test the package itself."""

import importlib.metadata

import streamcurvature as m


def test_version():
    assert importlib.metadata.version("streamcurvature") == m.__version__
