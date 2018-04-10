# coding=utf-8
'''
    inception v3 google 图片识别模型下载 并加载
'''
import tensorflow as tf
import os
import tarfile
import requests

inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# 下载 inception_model 模型
inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

print("finished, load: ", filename)
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

# TensorBoard log catalog
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# load inception graph
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()