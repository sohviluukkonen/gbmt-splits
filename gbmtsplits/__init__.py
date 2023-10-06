import os

from .split import GloballyBalancedSplit
from .clustering import *

__version__ = '0.0.5'
if os.path.exists(os.path.join(os.path.dirname(__file__), '_version.py')):
    from ._version import version
    __version__ = version

VERSION = __version__