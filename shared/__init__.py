"""Kernel-PE shared utilities."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("kernel-pe")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
