# coding=utf-8
import tensorflow as tf

# 构建 graph

input1 = tf.placeholder(tf.float64)
input2 = tf.placeholder(tf.float64)
# 数值乘法
new_value = tf.multiply(input1,input2)

# 自动关闭session功能
with tf.Session() as sess:
    #  feed_dict={X:x,Y:y,Z:z} 初始化变量功能
    print(sess.run(new_value,feed_dict={input1:23.0,input2:10.0}))


