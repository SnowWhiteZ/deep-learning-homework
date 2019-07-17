# import  numpy as np
# import tensorflow as tf
#
# def convert_to_one_hot(Y, C):
#     Y = np.eye(C)[Y.reshape(-1)].T
#     print(Y)
#     return Y
# array = np.array([1,3,6])
#
# sess = tf.Session()
# x = sess.run(tf.arg_max(array,0))
# print(x)
# #print(sess.run(tf.arg_max(array,1)))
# print(convert_to_one_hot(array,4))
# sess.close()
import numpy as np

import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

#%matplotlib inline #如果你使用的是jupyter notebook取消注释
#np.random.seed(1)

# y_hat = tf.constant(36,name='y_hat')
# y = tf.constant(39,name='y')
# loss = tf.Variable((y-y_hat)**2,name = 'loss')
#
# init = tf.global_variables_initializer()
#
#
#
# a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a,b)
# x = tf.placeholder(tf.int64,name='x')
#
# with tf.Session() as session:
#     # session.run(init)
#     print(session.run(2*x,feed_dict={x:4}))

def linear_function():
    np.random.seed(1)
    W = np.random.randn(4,3)
    X = np.random.randn(3,1)
    b = np.random.randn(4,1)

    Y = tf.add(tf.matmul(W,X),b)
    #Y = tf.add(np.dot(W,X) ,b)
    sess = tf.Session()
    result = sess.run(Y)
    sess.close()
    return result

#print("result = " +  str(linear_function()))

def sigmoid(z):
    x = tf.placeholder(tf.float32,name = 'x')

    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})

    return result

# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(12) = " + str(sigmoid(12)))


#使用独热编码
def one_hot_matrix(labels,C):
    C = tf.constant(C,name = 'C')

    one_hot_matrix = tf.one_hot(indices=labels,depth=C,axis=0)

    sess = tf.Session()

    one_hot = sess.run(one_hot_matrix)

    sess.close()
    return one_hot

# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels,C=4)
# print(str(one_hot))

#初始化0,1
def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    return  ones

# print ("ones = " + str(ones([3])))


    # 训练集：有从0到5的数字的1080张图片(64x64像素)，每个数字拥有180张图片。
    # 测试集：有从0到5的数字的120张图片(64x64像素)，每个数字拥有5张图片。
X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = tf_utils.load_dataset()
# index = 12
# plt.imshow(X_train_orig[index])
# plt.show()
# print("Y = " + str(np.squeeze(Y_train_orig[:,index])))
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
#每一列就是一个样本
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T
#归一化数据
X_train = X_train_flatten/255
X_test = X_test_flatten/255

print(Y_train_orig.shape)
Y_train = tf_utils.convert_to_one_hot(Y_train_orig,6)
print(Y_train.shape)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)
# print(X_train_orig.shape)(1080, 64, 64, 3)
# print(X_train_flatten.shape)(12288, 1080)

# print("训练集样本数 = " + str(X_train.shape[1]))
# print("测试集样本数 = " + str(X_test.shape[1]))
# print("X_train.shape: " + str(X_train.shape))
# print("Y_train.shape: " + str(Y_train.shape))
# print("X_test.shape: " + str(X_test.shape))
# print("Y_test.shape: " + str(Y_test.shape))
def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32,[n_x,None],name='X')#12288*
    Y = tf.placeholder(tf.float32,[n_y,None],name='Y')#6*
    keep_prob = tf.placeholder(tf.float32)
    return X,Y,keep_prob

# X, Y = create_placeholders(12288, 6)
# print("X = " + str(X))
# print("Y = " + str(Y))

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1',[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return parameters

# tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。
#
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))

#前向传播

def forward_propagation(X,parameters,keep_prob):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    L1 = tf.nn.dropout(A1,keep_prob)

    Z2 = tf.add(tf.matmul(W2,L1),b2)
    A2 = tf.nn.relu(Z2)
    L2 = tf.nn.dropout(A2,keep_prob)

    Z3 = tf.add(tf.matmul(W3,L2),b3)

    return Z3

# tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。
# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     print("Z3 = " + str(Z3))
def compute_cost(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

    return  cost

# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#     print("cost = " + str(cost))

def model(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,nums_epochs=1400,minibatch_size=32,print_cost=True,is_plot=True):

    #重新运行而不覆盖tf变量
    ops.reset_default_graph()

    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]

    costs = []

    X,Y,keep_prob = create_placeholders(n_x,n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X,parameters,keep_prob=0.9)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(nums_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed+1

            minibatches = tf_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:

                (minibatch_X,minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost],feed_dict={X:minibatch_X,Y:minibatch_Y,keep_prob:0.9})
                epoch_cost = epoch_cost+minibatch_cost/num_minibatches

            if epoch%5==0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        parameters = sess.run(parameters)
        print("参数已经保存")

        correct_prediction = tf.equal(tf.arg_max(Z3,0),tf.arg_max(Y,0))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

        train_acc = sess.run(accuracy,feed_dict={X:X_train,Y:Y_train,keep_prob:0.9})
        test_acc = sess.run(accuracy,feed_dict={X:X_test,Y:Y_test,keep_prob:1.0})
        print('train:'+str(train_acc))
        print('test'+str(test_acc))
        # print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        # print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

#开始时间
start_time = time.clock()
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.clock()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )
