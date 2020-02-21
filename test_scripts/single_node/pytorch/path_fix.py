import os
import sys

cur_folder = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(cur_folder, "../../"))

from logger import get_logger, log_time, sync_e # pylint: disable=import-error
