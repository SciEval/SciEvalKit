"""ChemBench dataset entry-point for VLMEvalKit."""
from loguru import logger

logger.disable("chembench")
__version__ = "0.3.0"

from .chembench import ChemBench

__all__ = ["ChemBench", "__version__"]
