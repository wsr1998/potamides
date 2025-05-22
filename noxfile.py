"""Configuration for Nox."""

import argparse
import shutil
from pathlib import Path

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()

# =============================================================================
# Linting


@session(uv_groups=["lint"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    precommit(s)  # reuse pre-commit session
    pylint(s)  # reuse pylint session


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run pre-commit."""
    s.run("pre-commit", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
def pylint(s: nox.Session, /) -> None:
    """Run PyLint."""
    s.run("pylint", "potamides", *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], reuse_venv=True)
def pytest(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.run("pytest", *s.posargs)


@session(uv_groups=["test"], reuse_venv=True)
def make_test_arraydiff(s: nox.Session, /) -> None:
    """
    Generate the `pytest-arraydiff` files.
    """
    s.run(
        "pytest",
        "-m",
        "array_compare",
        "--arraydiff-generate-path=tests/data",
        *s.posargs,
    )


@session(uv_groups=["test"], reuse_venv=True)
def make_test_mpl(s: nox.Session, /) -> None:
    """
    Generate the `pytest-mpl` baseline images.
    """
    s.run(
        "pytest",
        "-m",
        "mpl_image_compare",
        "--mpl-generate-hash-library=/Users/nmrs/local/potamides/tests/baseline/plot_hashes.json",
        *s.posargs,
    )


# =============================================================================
# Documentation


@session(uv_groups=["docs"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving. First positional argument is the target directory.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(s.posargs)
    serve = args.builder == "html" and s.interactive

    s.install("sphinx-autobuild")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )

    if serve:
        s.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        s.run("sphinx-build", "--keep-going", *shared_args)


@session(uv_groups=["docs"], reuse_venv=True)
def build_api_docs(s: nox.Session, /) -> None:
    """Build (regenerate) API docs."""
    s.install("sphinx")
    s.run(
        "sphinx-apidoc",
        "-o",
        "docs/api/",
        "--module-first",
        "--no-toc",
        "--force",
        "src/potamides",
    )


@session(uv_groups=["docs"], reuse_venv=True)
def build(s: nox.Session, /) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    s.install("build")
    s.run("python", "-m", "build")
