import tensorflow as tf
import time 
import collections
# from tensorflow.python.eager import profiler
from tensorflow.python.client import timeline
import json
from os.path import join, exists
from os import makedirs


class TimePair():
    def __init__(self):
        super().__init__()
        self.start = 0
        self.end = 0


class TFProfiler(tf.keras.callbacks.Callback):
    """
    """
    def __init__(self, log_path, run_metadata=None):
        super().__init__()
        self.batch_time = collections.defaultdict(TimePair)
        self.path = log_path
        self.predict_time = collections.defaultdict(TimePair)
        self.run_metadata = run_metadata
        self.steps_status = []

    def on_train_begin(self, logs=None):
        # profiler.start()
        pass
    
    def on_train_end(self, logs=None):
        # profiler.save(self.path, profiler.stop())
        pass

    def on_train_batch_begin(self, batch, logs=None):
        # print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
        self.batch_time[batch].start = time.time()
        
    def on_train_batch_end(self, batch, logs=None):
        # print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
        self.batch_time[batch].end = time.time()
        self.steps_status.append(timeline.Timeline(self.run_metadata.step_stats))

    def on_predict_batch_begin(self, batch, logs=None):
        """"""
        self.predict_time[batch].start = time.time()
    
    def on_predict_batch_end(self, batch, logs=None):
        """"""
        self.predict_time[batch].end = time.time()
        
    def summarize_training(self):
        interval = 0.0
        for bid in self.batch_time:
            interval += self.batch_time[bid].end - self.batch_time[bid].start
        return interval / len(self.batch_time)
    
    def writeout_traces(self):
        steps = []
        for event in self.steps_status:
            chrome_trace = event.generate_chrome_trace_format(show_dataflow=False)
            parsed_trace = json.loads(chrome_trace)
            steps.extend(parsed_trace['traceEvents'])
        
        if not exists(self.path):
            makedirs(self.path)
        with open(join(self.path, 'trace.json'), 'w') as ofile:
            json.dump({'traceEvents':steps}, ofile, indent=2)
    