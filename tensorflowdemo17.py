# coding=utf-8
'''
    重新训练模型的分类识别
    inception_v3 google 图片识别 （基于inception_v3模型 自定义自己的图片分类）
'''
import os
import tensorflow as tf
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

lines = tf.gfile.GFile('retrain/output_labels.txt').readlines()
uid_to_human = {}
for uid,line in enumerate(lines):
    line = line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]

# 创建一个图来存放Google训练好的模型
with tf.gfile.FastGFile('retrain/output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # 遍历目录
    for root, dirs, files in os.walk('retrain/images/'):
        for file in files:
            # load image
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            # image type : xxx.jpg
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            # 结果转成1维数据
            predictions = np.squeeze(predictions)
            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            print('image_path : '+image_path)

            # 排序 取5个值，置信度（desc）降序排序
            # top_k = predictions.argsort()[-5:][::-1]
            top_k = predictions.argsort()[-5:][::-1]
            print(top_k)
            for node_id in top_k:
                # 获取分类名称
                human_string = id_to_string(node_id)
                # 获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score*100))
            print('--------------------------')
            # 显示图片
            # img = Image.open(image_path)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()
