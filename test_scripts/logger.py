import time
import os
from datetime import datetime

class MyLogger:
    def __init__(self, logpath):
        self.log_file = open(logpath, 'w+')
    def info(self, msg):
        self.log_file.write(msg + '\n')
        # self.log_file.wrtie("\n")
    def __del__(self):
        self.log_file.close()

def get_logger(hvd):
    logdir = "~/horovod_logs/model_log"
    logdir = os.path.expanduser(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    dt = datetime.fromtimestamp(time.time())
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    logging_file = os.path.join(logdir, "model-{}-rank{}.log".format(timestamp, hvd.rank()))
    logger = MyLogger(logging_file)
    return logger