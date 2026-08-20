"""
Logging helpers.
"""
import logging
import os

ll = os.environ.get('MLT_LOG_LEVEL', default='INFO')

logging.basicConfig(
    # filename='log.log',
    level=getattr(logging, ll),
    format='%(levelname)s %(asctime)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger()