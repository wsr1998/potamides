from __future__ import annotations

import importlib.metadata

import streamcurvature as m


def test_version():
    assert importlib.metadata.version("streamcurvature") == m.__version__
