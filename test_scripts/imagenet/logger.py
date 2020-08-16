import time
import torch
import os
from datetime import datetime
import contextlib
import json

def sync_e():
    e = torch.cuda.Event()
    e.record()
    e.synchronize()

class MyLogger:
    def __init__(self, logpath):
        self.log_file = open(logpath, 'w+')
    def info(self, msg):
        self.log_file.write(msg + '\n')
        # self.log_file.wrtie("\n")
    def __del__(self):
        self.log_file.close()

def get_logger(id):
    logdir = "~/horovod_logs/model_log"
    logdir = os.path.expanduser(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    dt = datetime.fromtimestamp(time.time())
    timestamp = dt.strftime("%Y%m%d-%H%M%S")
    logging_file = os.path.join(logdir, "model-{}-rank{}.log".format(timestamp, id))
    logger = MyLogger(logging_file)
    return logger

@contextlib.contextmanager
def log_time(_logger : MyLogger, phase_name, id):
    lobj = {"ph": "X", "name": phase_name, "ts": time.time(), "pid": id, "dur": 0}
    yield
    lobj["dur"] = time.time()-lobj["ts"]
    _logger.info(json.dumps(lobj))
