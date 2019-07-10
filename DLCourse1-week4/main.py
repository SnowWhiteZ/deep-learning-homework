import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward #参见资料包
import lr_utils

np.random.seed(1)

#初始化参数 对于一个两层的神经网络结构而言，模型结构是线性->ReLU->线性->sigmod函数。
def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y,1))
    # 使用断言确保我的数据格式是正确的
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# print("==============测试initialize_parameters==============")
# parameters = initialize_parameters(3,2,1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def initialize_parameters_deep(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1])/np.sqrt(layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        #确保我要的数据的格式是正确的
        assert(parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters

# #测试initialize_parameters_deep
# print("==============测试initialize_parameters_deep==============")
# layers_dims = [5,4,3]
# parameters = initialize_parameters_deep(layers_dims)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

#前向传播部分
def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    assert (Z.shape ==(W.shape[0],A.shape[1]))

    cache = (A,W,b)
    return Z,cache

# #测试linear_forward
# print("==============测试linear_forward==============")
# A,W,b = testCases.linear_forward_test_case()
# Z,linear_cache = linear_forward(A,W,b)
# print("Z = " + str(Z))

#线性+激活
def linear_activation_forward(A_prev,W,b,activation):
    if activation=="sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)

    cache = (linear_cache,activation_cache)

    return A,cache

#测试linear_activation_forward
# print("==============测试linear_activation_forward==============")
# A_prev, W,b = testCases.linear_activation_forward_test_case()
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("sigmoid，A = " + str(A))
#
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("ReLU，A = " + str(A))


def L_model_forward(X,parameters):
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation='relu')
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')

    caches.append(cache)

    return AL,caches

#测试L_model_forward
# print("==============测试L_model_forward==============")
# X,parameters = testCases.L_model_forward_test_case()
# AL,caches = L_model_forward(X,parameters)
# print("AL = " + str(AL))
# print("caches 的长度为 = " + str(len(caches)))

def compute_cost(AL,Y):
    m = Y.shape[1]  #数量
    cost = -np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m
    cost = np.squeeze(cost)

    assert (cost.shape==())

    return cost

# #测试compute_cost
# print("==============测试compute_cost==============")
# Y,AL = testCases.compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))


def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev  = np.dot(W.T,dZ)

    return dA_prev,dW,db
##测试linear_backward
# print("==============测试linear_backward==============")
# dZ, linear_cache = testCases.linear_backward_test_case()
#
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

def linear_activation_backward(dA,cache,activation='relu'):
    linear_cache,activation_cache = cache
    if activation=='relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev ,dW,db = linear_backward(dZ,linear_cache)
    elif activation=='sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db

#测试linear_activation_backward
# print("==============测试linear_activation_backward==============")
# AL, linear_activation_cache = testCases.linear_activation_backward_test_case()
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

#构建多层模型向后传播：
def L_model_backward(AL,Y,caches):

    grads = {}
    L = len(caches)
    m = Y.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache = caches[L-1]
    grads['dA'+str(L)],grads['dW'+str(L)],grads['db'+str(L)] = linear_activation_backward(dAL,current_cache,'sigmoid')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads['dA'+str(l+2)],current_cache ,'relu')
        grads['dA'+str(l+1)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp
    return grads

#测试L_model_backward
# print("==============测试L_model_backward==============")
# AL, Y_assess, caches = testCases.L_model_backward_test_case()
# grads = L_model_backward(AL, Y_assess, caches)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dA1 = "+ str(grads["dA1"]))

## update parameters
def update_parameters(parameters,grads,learning_rate):
    L=len(parameters)//2#
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] -learning_rate*grads['dW'+str(l+1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters

#测试update_parameters
# print("==============测试update_parameters==============")
# parameters, grads = testCases.update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.1)
#
# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))


# ready to build two layers nn
def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,epoch = 3000,print_cost = False,isPlot = True):
    np.random.seed(1)
    grads = {}
    costs = []
    (n_x,n_h,n_y) = layers_dims

    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for  i in range(epoch):

        A1,cache1= linear_activation_forward(X,W1,b1,'relu')
        A2,cache2 = linear_activation_forward(A1,W2,b2,'sigmoid')

        cost  = compute_cost(A2,Y)

        dA2 = -(np.divide(Y,A2)-np.divide(1-Y,1-A2))

        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,'sigmoid')
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters,grads,learning_rate)
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']


        if i %100==0:
            costs.append(cost)
            if print_cost:
                print("第", i ,"次迭代，成本值为：" ,np.squeeze(cost))
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterstions(per tens')
        plt.title('Lraening rate='+str(learning_rate))
        plt.show()
    return  parameters

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

# n_x = 12288
# n_h = 7
# n_y = 1
# # layers_dims = (n_x,n_h,n_y)
# #
# # parameters1 = two_layer_model(train_x, train_set_y, layers_dims = (n_x, n_h, n_y), epoch = 2500, print_cost=True,isPlot=True)
# # parameters = parameters1
def predict(X,y,parameters):
    m = X.shape[1]
    n = len(parameters)//2
    p = np.zeros((1,m))
    probas,caches = L_model_forward(X,parameters)

    for i in range(0,m):
        if probas[0,i]>0.5:
            p[0,i] = 1
        else:
            p[0,i]=0
    print('accuracy:'+str(float(np.sum(p==y))/m))

    return p

# prediction_train = predict(train_x,train_y,parameters)
# prediction_test = predict(test_x,test_y,parameters)



#########  build n layers nn:
def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,epoch=3000,print_cost = False,isPlot = True):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0,epoch):
        AL ,caches = L_model_forward(X,parameters)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        parameters = update_parameters(parameters,grads,learning_rate)

        if i%100==0:
            costs.append(cost)
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))

    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per tens)')
        plt.title('learning rate ='+str(learning_rate))
        plt.show()
    return parameters

layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, epoch = 2500, print_cost = True,isPlot=True)

pred_train = predict(train_x, train_y, parameters) #训练集
pred_test = predict(test_x, test_y, parameters) #测试集

def print_mislabeled_images(classes, X, y, p):
    """
    绘制预测和实际不同的图像。
        X - 数据集
        y - 实际的标签
        p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


print_mislabeled_images(classes, test_x, test_y, pred_test)

