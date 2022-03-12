import importlib.util
import logging

from .utils import importlib_metadata


logger = logging.getLogger(__name__)


def is_torch_available():
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False

    return _torch_available
