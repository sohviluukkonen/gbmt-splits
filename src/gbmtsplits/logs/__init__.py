import logging
import sys

logger = None

if not logger:
    logger = logging.getLogger("gbmtsplits")
    logger.setLevel(logging.INFO)


def setLogger(log):
    sys.modules[__name__].gbmtsplits = log