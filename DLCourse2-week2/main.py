# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils
import testCase


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters,grads,learning_rate):


    L = len(parameters) // 2 #神经网络的层数

    #更新每个参数
    for l in range(L):
        parameters["W" + str(l +1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l +1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

# #测试update_parameters_with_gd
# print("-------------测试update_parameters_with_gd-------------")
# parameters , grads , learning_rate = testCase.update_parameters_with_gd_test_case()
# parameters = update_parameters_with_gd(parameters,grads,learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def random_mini_batches(X,Y,mini_batch_size = 64,seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    #打乱顺序：
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1,m))


    #第二部，分割
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m%mini_batch_size!=0:
        mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


#测试random_mini_batches
# print("-------------测试random_mini_batches-------------")
# X_assess,Y_assess,mini_batch_size = testCase.random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess,Y_assess,mini_batch_size)
#
# print("第1个mini_batch_X 的维度为：",mini_batches[0][0].shape)
# print("第1个mini_batch_Y 的维度为：",mini_batches[0][1].shape)
# print("第2个mini_batch_X 的维度为：",mini_batches[1][0].shape)
# print("第2个mini_batch_Y 的维度为：",mini_batches[1][1].shape)
# print("第3个mini_batch_X 的维度为：",mini_batches[2][0].shape)
# print("第3个mini_batch_Y 的维度为：",mini_batches[2][1].shape)

#初始化动量
def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}

    for l in range(L):
        v['dW'+str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])
        v['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])
    return v

#测试initialize_velocity
# print("-------------测试initialize_velocity-------------")
# parameters = testCase.initialize_velocity_test_case()
# v = initialize_velocity(parameters)
#
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))

def update_parameters_with_momentun(parameters,grads,v,beta,learning_rate):
    L = len(parameters) // 2
    for l in range(L):  # 计算速度
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)] #更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
    return parameters,v


#测试update_parameters_with_momentun
# print("-------------测试update_parameters_with_momentun-------------")
# parameters,grads,v = testCase.update_parameters_with_momentum_test_case()
# update_parameters_with_momentun(parameters,grads,v,beta=0.9,learning_rate=0.01)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))

#adam算法

def initialize_adam(parameters):
    """
    初始化v和s，它们都是字典类型的变量，都包含了以下字段：
        - keys: "dW1", "db1", ..., "dWL", "dbL"
        - values：与对应的梯度/参数相同维度的值为零的numpy矩阵

    参数：
        parameters - 包含了以下参数的字典变量：
            parameters["W" + str(l)] = Wl
            parameters["b" + str(l)] = bl
    返回：
        v - 包含梯度的指数加权平均值，字段如下：
            v["dW" + str(l)] = ...
            v["db" + str(l)] = ...
        s - 包含平方梯度的指数加权平均值，字段如下：
            s["dW" + str(l)] = ...
            s["db" + str(l)] = ...

    """

    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return (v,s)


def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1 = 0.9,beta2 = 0.999,epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {} #偏差修正后的值
    s_corrected = {} #偏差修正后的值

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1,t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1,t))
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])
        #计算第二阶段的偏差修正后的估计值，输入："s , beta2 , t"，输出："s_corrected"
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2,t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2,t))
        #更新参数，输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon". 输出: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))
    return (parameters,v,s)

#测试update_with_parameters_with_adam
# print("-------------测试update_with_parameters_with_adam-------------")
# parameters , grads , v , s = testCase.update_parameters_with_adam_test_case()
# update_parameters_with_adam(parameters,grads,v,s,t=2)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))

train_X, train_Y = opt_utils.load_dataset(is_plot=True)


def model(X,Y,layer_dims,optimizer,learning_rate=0.0007,mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,
          epsilon=1e-8,num_epochs=10000,print_cost=True,is_plot=True):
    L = len(layer_dims)
    costs = []
    t=0
    seed= 10


    paramaters = opt_utils.initialize_parameters(layer_dims)

    if optimizer=='gd':
        pass
    elif optimizer=='momentum':
        v= initialize_velocity(paramaters)
    elif optimizer=='adam':

        v,s = initialize_adam(paramaters)

    else:
        print('error')
        exit(1)
    for i in range(num_epochs):
        seed = seed+1
        minibatches = random_mini_batches(X,Y,mini_batch_size,seed)

        for minibatch in minibatches:
            (minibatch_X,minibatch_Y) = minibatch

            A3 ,cache = opt_utils.forward_propagation(minibatch_X,paramaters)

            cost = opt_utils.compute_cost(A3,minibatch_Y)

            grads = opt_utils.backward_propagation(minibatch_X,minibatch_Y,cache)

            if optimizer=='gd':
                paramaters = update_parameters_with_gd(paramaters,grads,learning_rate)
            elif optimizer=='momentum':
                paramaters,v = update_parameters_with_momentun(paramaters,grads,v,beta1,learning_rate)
            elif optimizer=='adam':
                t = t + 1
                paramaters,v,s = update_parameters_with_adam(paramaters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
        if i %100==0:
            costs.append(cost)
            if print_cost and i%1000==0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
                # 是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
    return paramaters

#使用普通的梯度下降
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X, train_Y, layers_dims, optimizer="adam",is_plot=True)

#预测
preditions = opt_utils.predict(train_X,train_Y,parameters)

#绘制分类图
plt.title("Model with adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)