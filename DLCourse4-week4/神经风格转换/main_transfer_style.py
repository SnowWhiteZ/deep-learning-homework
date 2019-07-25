import time
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import nst_utils
import numpy as np
import tensorflow as tf

#%matplotlib inline

# model = nst_utils.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
#
# print(model)
#
# content_image = scipy.misc.imread("images/louvre.jpg")
# imshow(content_image)
# plt.show()


#计算内容损失
def compute_content_cost(a_C,a_G):
    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C,[n_H*n_W,n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))

    J_content = 1/(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))
    return J_content

# tf.reset_default_graph()
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_content = compute_content_cost(a_C, a_G)
#     print("J_content = " + str(J_content.eval()))
#
#     test.close()



#计算风格矩阵G
def gram_matrix(A):

    GA = tf.matmul(A,A,transpose_b=True)
    return GA
# tf.reset_default_graph()
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     A = tf.random_normal([3, 2*1], mean=1, stddev=4)
#     GA = gram_matrix(A)
#
#     print("GA = " + str(GA.eval()))
#
#     test.close()

#计算风格损失
def compute_layer_style_cost(a_S,a_G):

    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W,n_C]))
    a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))

    #计算风格矩阵
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    #compute style loss
    J_style_layer = 1/(4*n_C*n_C*n_H*n_H*n_W*n_W)*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))

    return J_style_layer

# tf.reset_default_graph()
#
# with tf.Session() as test:
#     tf.set_random_seed(1)
#     a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
#     J_style_layer = compute_layer_style_cost(a_S, a_G)
#
#     print("J_style_layer = " + str(J_style_layer.eval()))
#
#     test.close()

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]
# sess = tf.Session()

def compute_style_cost(model,STYLE_LAYERS):
    J_style = 0

    for layer_name,coeff in STYLE_LAYERS:
        out = model[layer_name]

        a_S = sess.run(out)

        a_G = out

        J_style_layer = compute_layer_style_cost(a_S,a_G)

        J_style += coeff *J_style_layer

    return  J_style

def total_cost(J_content,J_style,alpha = 10,beta = 40):
    J = alpha*J_content + beta * J_style
    return J


# tf.reset_default_graph()
#
# with tf.Session() as test:
#     np.random.seed(3)
#     J_content = np.random.randn()
#     J_style = np.random.randn()
#     J = total_cost(J_content, J_style)
#     print("J = " + str(J))
#
#     test.close()

#重设图
tf.reset_default_graph()

#第1步：创建交互会话
sess = tf.InteractiveSession()

#第2步：加载内容图像(卢浮宫博物馆图片),并归一化图像
content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = nst_utils.reshape_and_normalize_image(content_image)

#第3步：加载风格图像(印象派的风格),并归一化图像
style_image = scipy.misc.imread("images/monet.jpg")
style_image = nst_utils.reshape_and_normalize_image(style_image)

#第4步：随机初始化生成的图像,通过在内容图像中添加随机噪声来产生噪声图像
generated_image = nst_utils.generate_noise_image(content_image)
imshow(generated_image[0])
plt.show()

#第5步：加载VGG16模型
model = nst_utils.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

sess.run(model['input'].assign(content_image))

out = model['conv4_2']

a_C = sess.run(out)

a_G = out

J_content = compute_content_cost(a_C,a_G)

sess.run(model['input'].assign(style_image))

J_style = compute_style_cost(model,STYLE_LAYERS)

J= total_cost(J_content,J_style,alpha=10,beta=40)

optimizer = tf.train.AdamOptimizer(2.0)

train_step = optimizer.minimize(J)

def model_nn(sess,input_image,num_iterations=200,is_print_info=True,is_plot=True,is_save_process_image=True,save_last_image_to='output/generated_image.jpg'):
    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):
        sess.run(train_step)

        generated_image  = sess.run(model['input'])

        if is_print_info and i%20==0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("第 " + str(i) + "轮训练," + "  总成本为:" + str(Jt) + "  内容成本为：" + str(Jc) + "  风格成本为：" + str(Js))
        if is_save_process_image:
            nst_utils.save_image("output/" + str(i) + ".png", generated_image)
    nst_utils.save_image(save_last_image_to, generated_image)
    return generated_image


#开始时间
start_time = time.clock()

#非GPU版本,约25-30min
generated_image = model_nn(sess, generated_image)


#使用GPU，约1-2min
# with tf.device("/gpu:0"):
#     generated_image = model_nn(sess, generated_image)

#结束时间
end_time = time.clock()

#计算时差
minium = end_time - start_time

print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")



