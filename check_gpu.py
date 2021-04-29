import tensorflow as tf

with tf.compat.v1.Session() as sess:
    devices = sess.list_devices()
    print(devices)
