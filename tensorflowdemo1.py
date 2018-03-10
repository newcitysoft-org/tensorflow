# coding=utf-8
import tensorflow as tf

# 创建 graph 2个矩阵value
v1 = tf.constant([[2,3]])
v2 = tf.constant([[2],[3]])
# 矩阵乘法
product = tf.matmul(v1,v2)

# 开启执行环境session
sess = tf.Session()
# 运行 graph
result = sess.run(product)
print(result)
# 关闭环境
sess.close()