# coding=utf-8
import tensorflow as tf

hello = tf.constant('Hello,TensorFlow!')
with tf.Session() as sess:
    print (sess.run(hello))