import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import horovod.tensorflow.keras as hvd
from datetime import datetime

# Horovod: initialize Horovod.
hvd.init()

# =============== load data mnist
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Input image dimensions
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# --------------- end data loading

# =============== Define models
model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

opt = keras.optimizers.Adadelta(1.0 * hvd.size())
# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata= tf.RunMetadata()

log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

# ================ compile models
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
            #   options=run_options, 
            #   run_metadata=run_metadata,
              )

# Training
model.fit(x_train, y_train, epochs=1,
        callbacks=[tensorboard_callback])

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

run_metadata= tf.RunMetadata()
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('timelines/tf_redo_getmeta.json', 'w') as f:
    f.write(ctf)