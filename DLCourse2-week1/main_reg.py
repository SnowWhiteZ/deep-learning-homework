# 每一个点代表球落下的可能的位置，蓝色代表己方的球员会抢到球，红色代表对手的球员会抢到球，我们要做的就是使用模型来画出一条线，来找到适合我方球员能抢到球的位置。
# 我们要做以下三件事，来对比出不同的模型的优劣：
#
#     不使用正则化
#     使用正则化
#     2.1 使用L2正则化
#     2.2 使用随机节点删除


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils   #第一部分，初始化
import reg_utils    #第二部分，正则化
import gc_utils     #第三部分，梯度校验
#%matplotlib inline #如果你使用的是Jupyter Notebook，请取消注释。
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=True)

def model(X,Y,learning_rate = 0.3,num_iterations = 30000,print_cost = True,is_plot = True,lambd = 0,keep_prob=1):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]

    parameters = reg_utils.initialize_parameters(layers_dims)

    for i in range(num_iterations):

        #forward propagation
        if keep_prob==1:
            a3 ,cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob<1:
            a3,cache = forward_propagation_with_dropout(X,parameters,keep_prob=0.5)
        else:
            print('error')
            exit()
        #是否使用L2正则：
        if lambd ==0:
            cost = reg_utils.compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)

        ##反向传播：

        if lambd==0 and keep_prob==1 :
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd!=0:
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob<1:
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)

        parameters = reg_utils.update_parameters(parameters,grads,learning_rate)

        if i%1000==0:
            costs.append(cost)
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    if is_plot:
        plt.plot(costs)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters



def compute_cost_with_regularization(A3, Y, parameters,lambd):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    cross_entropy_cost = reg_utils.compute_cost(A3,Y)
    L2_regularization_cost = lambd * (np.sum(np.square(W1))  + np.sum(np.square(W2))  +np.sum(np.square(W3)))/(2*m)
    cost = cross_entropy_cost+L2_regularization_cost
    return cost



def backward_propagation_with_regularization(X,Y,cache,lambd):
    m = X.shape[1]

    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 =A3-Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients


def forward_propagation_with_dropout(X,parameters,keep_prob = 0.5):
    """
        实现具有随机舍弃节点的前向传播。
        LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

        参数：
            X  - 输入数据集，维度为（2，示例数）
            parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
                W1  - 权重矩阵，维度为（20,2）
                b1  - 偏向量，维度为（20,1）
                W2  - 权重矩阵，维度为（3,20）
                b2  - 偏向量，维度为（3,1）
                W3  - 权重矩阵，维度为（1,3）
                b3  - 偏向量，维度为（1,1）
            keep_prob  - 随机删除的概率，实数
        返回：
            A3  - 最后的激活值，维度为（1,1），正向传播的输出
            cache - 存储了一些用于计算反向传播的数值的元组
        """

    np.random.seed(1)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1,X)+b1
    A1 = reg_utils.relu(Z1)

    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1<keep_prob
    A1 = A1 *D1
    A1 = A1/keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = reg_utils.relu(Z2)

    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = reg_utils.sigmoid(Z3)
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache




def backward_propagation_with_dropout(X,Y,cache,keep_prob):

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = dA2 * D2  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 / keep_prob # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1 # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA1 = dA1 / keep_prob # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients


parameters = model(train_X, train_Y,keep_prob=0.86,learning_rate=0.3,is_plot=True)
print("训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)