"""Doctest configuration."""

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

import jax
from sybil import Sybil
from sybil.parsers import myst, rest
from sybil.sybil import SybilCollection

jax.config.update("jax_enable_x64", True)

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
)
rst_docs = Sybil(  # TODO: deprecate
    parsers=[
        rest.DocTestParser(optionflags=optionflags),
        rest.PythonCodeBlockParser(),
        rest.SkipParser(),
    ],
    patterns=["*.rst", "*.py"],
)

pytest_collect_file = SybilCollection((docs, python, rst_docs)).pytest()
