"""Doctest configuration."""

import builtins
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from typing import Any

import jax
import numpy as np
import pytest
from sybil import Sybil
from sybil.parsers import myst, rest
from sybil.sybil import SybilCollection

jax.config.update("jax_enable_x64", True)

###########################################################
# Pytest fixtures


def _custom_repr(x: Any, /) -> Any:
    if isinstance(x, jax.Array):
        x = x.copy()
        x = x.at[x == 0].set(np.zeros((), dtype=x.dtype))
    return x


@pytest.fixture(scope="session")
def normalize_negative_zero():
    """
    Ensure that -0.0 is displayed as 0.0 in all jax output.
    Applied automatically to all tests, including doctests.
    """
    orig_print = builtins.print

    def _custom_print(*args: Any, **kwargs: Any) -> None:
        new_args = [_custom_repr(a) for a in args]
        return orig_print(*new_args, **kwargs)

    builtins.print = _custom_print
    yield
    builtins.print = orig_print


###########################################################
# Sybil Doctest Configuration

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers = [
    myst.DocTestDirectiveParser(optionflags=optionflags),
    myst.PythonCodeBlockParser(doctest_optionflags=optionflags),
    myst.SkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(  # TODO: get working with myst parsers
    parsers=[
        rest.DocTestParser(optionflags=optionflags),
        rest.PythonCodeBlockParser(),
        rest.SkipParser(),
    ],
    patterns=["*.py"],
    fixtures=["normalize_negative_zero"],
)
rst_docs = Sybil(  # TODO: deprecate
    parsers=[
        rest.DocTestParser(optionflags=optionflags),
        rest.PythonCodeBlockParser(),
        rest.SkipParser(),
    ],
    patterns=["*.rst"],
    fixtures=["normalize_negative_zero"],
)

pytest_collect_file = SybilCollection((docs, python, rst_docs)).pytest()


###########################################################
# Pytest configuration


def pytest_collection_modifyitems(items) -> None:
    """
    Automatically modify pytest items to suppress return-value warnings for
    array_compare tests.

    Tests marked with `@pytest.mark.array_compare` are expected to return arrays
    for comparison, which violates pytest's default expectation that tests
    return `None`. This function adds a warning filter to such tests to ignore
    the PytestReturnNotNoneWarning.

    Parameters
    ----------
    items : list
        A list of pytest Item objects collected during test discovery.

    """
    for item in items:
        if "array_compare" in item.keywords or "as_numpy" in item.keywords:
            item.add_marker(
                pytest.mark.filterwarnings("ignore::pytest.PytestReturnNotNoneWarning")
            )


def pytest_runtest_call(item) -> None:
    # Need to wrap the test function to ensure it returns a numpy array since
    # pytest-array only works with numpy
    if item.get_closest_marker("array_compare"):
        test_func = item.obj

        def wrapped(*args: Any, **kwargs: Any) -> np.ndarray:
            result = test_func(*args, **kwargs)
            return np.array(result)

        item.obj = wrapped
