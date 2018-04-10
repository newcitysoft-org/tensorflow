# coding:utf-8
'''
    inception_v3 google 基于摄像头的图像识别
'''
import cv2
import time
import threading
import os
import Queue
import numpy as np
import tensorflow as tf
#-----------------------------------start 参数配置----------------------------------------------
num = 0 #人脸分析延迟初始化
numfacetime = 1
ISOTIMEFORMAT='%Y%m%d%H%M%S' #时间格式


path = "pic" #图片读取路径（根目录/pic/*.jpg）
q = Queue.Queue() #线程信息共享
sess = tf.Session()
#-----------------------------------end 参数配置---------------------------------------------
class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, img_facename,user,q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.img_facename = img_facename
        self.user = user
        self.q = q
    def run(self):#把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        msg =  getProcess(self.img_facename, self.user)
        self.q.put(msg)

class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n******对应的分类名称文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        # 行为单位读取
        for line in proto_as_ascii_lines:
            # 去掉换行符
            line = line.strip('\n')
            # 按照'\t'分割
            parsed_items = line.split('\t')
            # 获得分类编号
            uid = parsed_items[0]
            # 获得分类名称
            human_string = parsed_items[1]
            # 保存编号字符串n*****与分类名称映射关系
            uid_to_human[uid] = human_string

        # 加载分类字符串 对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                # 获取分类编号1-1000
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                # 获得分类编号字符串
                target_class_string = line.split(': ')[1]
                # save map
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # 建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            # 获取分类名称
            name = uid_to_human[val]
            # 建立分类编号1-1000的分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# 创建一个图来存放Google训练好的模型
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

node_lookup = NodeLookup()

def getProcess(img_facename, user):
    msg = ""
    try :
        # 遍历目录
        for root, dirs, files in os.walk('pic/'):
            # load image
            image_ = os.path.join(root, files[0])
            image_data = tf.gfile.FastGFile(image_, 'rb').read()
            # image type : xxx.jpg
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data}  )
            predictions = np.squeeze(predictions)
            # 打印图片路径及名称
            image_path = os.path.join(root, files[0])
            # print('image_path : '+image_path)
            # 排序 取1个值，置信度（desc）降序排序
            top_k = predictions.argsort()[-1:][::-1]
            for node_id in top_k:
                # 获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                # 获取该分类的置信度
                score = predictions[node_id]
                # 取置信度最高组的位置[0]的名称
                # msg = str(human_string.split(',')[0])+';score='+str(score)
                # 分词切割，默认显示第一个单词
                msg = str('%s (score=%.5f)' % (human_string.split(',')[0], score))
                # print('%s (score=%.5f)' % (human_string, score))
        # print "end->>",str(img_facename)
    except Exception as e:
        # print e
        msg = str(e)
    return msg




cv2.namedWindow("test")#命名一个窗口
cap=cv2.VideoCapture(0)#打开0号摄像头
success, frame = cap.read()#读取一桢图像，前一个返回值是是否成功，后一个返回值是图像本身
rmsg = "..."
while success:
    success, frame = cap.read()
    size=frame.shape[:2]#获得当前桢彩色图像的大小
    image=np.zeros(size,dtype=np.float16)#定义一个与当前桢图像大小相同的的灰度图像矩阵
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#将当前桢图像转换成灰度图像（这里有修改）
    cv2.equalizeHist(image, image)#灰度图像进行直方图等距化
    user = 'img'
    img_name = path+'/'+user+'.jpg'
    cv2.imwrite(img_name,frame)
    try:

        # 线程模式 创建新线程

        # thread1 = myThread(1, "Thread-1",img_name,user,q)

        # 开启线程
        # thread1.start()

        #非线程模式
        rmsg = getProcess(img_name, user)
    except:
        print "Error: unable to start thread"
    # 线程模式:

    # if not q.empty():
    #     rmsg = q.get()

    print rmsg
    cv2.putText(frame,rmsg,(0, 25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
    cv2.imshow("test", frame)#显示图像
    key=cv2.waitKey(10)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break
cv2.destroyWindow("test")
