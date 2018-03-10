# coding=utf-8
import tensorflow as tf

# 创建 graph

num = tf.Variable(0,name="count")
# 创建加法 步长10
new_value = tf.add(num,10)
# 赋值操作，把 new_value 赋给 num
op = tf.assign(num,new_value)

# 自动关闭session功能
with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())
    print("init value:"+str(sess.run(num)))
    for i in range(10):
        # 运行 op
        sess.run(op)
        print(sess.run(num))


