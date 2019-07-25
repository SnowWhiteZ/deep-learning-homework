from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

#------------用于绘制模型细节，可选--------------#
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#------------------------------------------------#

K.set_image_data_format('channels_first')

import time
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import fr_utils
from inception_blocks_v2 import *

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

#np.set_printoptions(threshold=np.nan)


FRmodel = faceRecoModel(input_shape=(3,96,96))

#print("参数数量"+str(FRmodel.count_params()))


#实现三元损失
def triplet_loss(y_true,y_pred,alpha=0.2):
    #获取目标，正负例的128维向量表示
    anchor,positive,negative = y_pred[0],y_pred[1],y_pred[2]
    pos_list = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_list = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)

    basic_loss = tf.add(tf.subtract(pos_list,neg_list),alpha)

    loss = tf.reduce_sum(tf.maximum(basic_loss,0))

    return  loss
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     y_true = (None, None, None)
#     y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
#               tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
#               tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
#     loss = triplet_loss(y_true, y_pred)
#
#     print("loss = " + str(loss.eval()))

start_time = time.clock()
FRmodel.compile(optimizer='adam',loss=triplet_loss,metrics = ['accuracy'])

fr_utils.load_weights_from_FaceNet(FRmodel)

end_time = time.clock()

minium = end_time-start_time

print("执行了"+str(int(minium/60))+'分'+str(int(minium%60))+'秒')

database = {}
database["danielle"] = fr_utils.img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = fr_utils.img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = fr_utils.img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = fr_utils.img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = fr_utils.img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = fr_utils.img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = fr_utils.img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = fr_utils.img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = fr_utils.img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = fr_utils.img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = fr_utils.img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = fr_utils.img_to_encoding("images/arnaud.jpg", FRmodel)


def verify(image_path, identity, database, model):
    """
    对“identity”与“image_path”的编码进行验证。

    参数：
        image_path -- 摄像头的图片。
        identity -- 字符类型，想要验证的人的名字。
        database -- 字典类型，包含了成员的名字信息与对应的编码。
        model -- 在Keras的模型的实例。

    返回：
        dist -- 摄像头的图片与数据库中的图片的编码的差距。
        is_open_door -- boolean,是否该开门。
    """
    #第一步：计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path, model)

    #第二步：计算与数据库中保存的编码的差距
    dist = np.linalg.norm(encoding - database[identity])

    #第三步：判断是否打开门
    if dist < 0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与" + str(identity) + "不符！")
        is_door_open = False

    return dist, is_door_open

verify("images/camera_0.jpg","younes",database,FRmodel)

verify("images/camera_2.jpg", "kian", database, FRmodel)

#人脸识别系统是一对多的关系，需要找寻数据库中存在的所有的数据查询
def who_is_it(image_path,database,model):
    encoding = fr_utils.img_to_encoding(image_path,model)

    min_dist = 100

    for(name,db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        if dist<min_dist:
            min_dist = dist
            identity  =name
    if min_dist>0.7:
        print("查无此人")
    else:
        print("姓名："+str(identity)+"差距:"+str(min_dist))
    return min_dist,identity
who_is_it("images/camera_0.jpg", database, FRmodel)

