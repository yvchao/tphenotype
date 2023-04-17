import sys

from .model import JointPredictor, LaplaceEncoder, Predictor

if sys.version_info[:2] >= (3, 8):  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version
else:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    # Change here if project is renamed and does not equal the package name
    import_name = "tphenotype"
    dist_name = "tphenotype"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = ["LaplaceEncoder", "Predictor", "JointPredictor"]
