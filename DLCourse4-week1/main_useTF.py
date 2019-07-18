##########使用TensorFlow框架实现了一下~

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

import cnn_utils

#%matplotlib inline
np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
# index = 6
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print ("y = " + str(np.squeeze(train_set_y_orig[:, index])))
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

def create_placeholders(n_H0,n_W0,n_C0,n_y):
    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])

    return  X,Y

# X , Y = create_placeholders(64,64,3,6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))

def initialize_parameters():

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
#
#     sess_test.close()

def forward_propagation(X,parameters):
    W1 = parameters['W1'] *np.sqrt(2)
    W2 = parameters['W2'] * np.sqrt(2/196)

    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')

    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

    P = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P,6,activation_fn = None)

    return Z3

# tf.reset_default_graph()
# np.random.seed(1)
#
# with tf.Session() as sess_test:
#     X,Y = create_placeholders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#
#     a = sess_test.run(Z3,{X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#     print("Z3 = " + str(a))
#
#     sess_test.close()


def compute_cost(Z3,Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

    return  cost


def model(X_train,Y_train,X_test,Y_test,learning_rate=0.005,num_epoch=1500,minibatch_size=64,print_cost =True,is_plot=True):
    ops.reset_default_graph()

    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X,Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    parameters = initialize_parameters()

    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epoch):

            minibatch_cost=0
            num_minibatches = int(m/minibatch_size)

            seed = seed+1

            minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _,temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

                minibatch_cost+=temp_cost/num_minibatches
            if print_cost:
                if epoch%100==0:
                    print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

            if epoch % 10 == 0:
                costs.append(minibatch_cost)
        # 数据处理完毕，绘制成本曲线
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show() #开始预测数据
            # ## 计算当前的预测情况
        predict_op = tf.arg_max(Z3,1)
        corrent_prediction = tf.equal(predict_op , tf.arg_max(Y,1))
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))
        print("corrent_prediction accuracy= " + str(accuracy))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuary = accuracy.eval({X: X_test, Y: Y_test})
        print("训练集准确度：" + str(train_accuracy))
        print("测试集准确度：" + str(test_accuary))
        return (train_accuracy,test_accuary,parameters)


_, _, parameters = model(X_train, Y_train, X_test, Y_test,num_epoch=1500)