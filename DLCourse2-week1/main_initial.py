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

train_X,train_Y,test_X,test_Y = init_utils.load_dataset(is_plot=True)


def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W'+str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))
    return parameters

def intialize_parameters_random(layers_dims):
    parameters = {}

    L = len(layers_dims)

    for l in range(1,L):
        parameters['W'+str(l)] = np.random.rand(layers_dims[l],layers_dims[l-1])*10
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))

    return  parameters

# def initialize_parameters_he(layers_dims):
#     """
#     参数：
#         layers_dims - 列表，模型的层数和对应每一层的节点的数量
#     返回
#         parameters - 包含了所有W和b的字典
#             W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
#             b1 - 偏置向量，维度为（layers_dims[1],1）
#             ···
#             WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
#             b1 - 偏置向量，维度为（layers_dims[L],1）
#     """
#
#     np.random.seed(3)               # 指定随机种子
#     parameters = {}
#     L = len(layers_dims)            # 层数
#
#     for l in range(1, L):
#         parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
#         parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
#
#         #使用断言确保我的数据格式是正确的
#         assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
#         assert(parameters["b" + str(l)].shape == (layers_dims[l],1))
#
#     return parameters

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}

    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def model(X,Y,learning_rate = 0.01,num_iterations = 15000,print_cost = True,initialization = "he",is_polt=True):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]

    if initialization=='zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization =='random':
        parameters = intialize_parameters_random(layers_dims)

    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    else:
        print('error')
        exit()
    for i in range(num_iterations):
        a3,cache = init_utils.forward_propagation(X,parameters)
        cost = init_utils.compute_loss(a3,Y)
        grads = init_utils.backward_propagation(X,Y,cache)
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)

        if i%1000==0:
            costs.append(cost)
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
#parameters = model(train_X, train_Y, initialization = "random",is_polt=True)
parameters = model(train_X, train_Y, initialization = "he",is_polt=True)
print ("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print ("测试集:")
predictions_test = init_utils.predict(test_X, test_Y, parameters)
print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

