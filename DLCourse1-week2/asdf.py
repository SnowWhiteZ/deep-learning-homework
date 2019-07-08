import numpy as np
import matplotlib.pyplot as plt
import h5py
import pylab
from lr_utils import load_dataset
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
index=24
#plt.imshow(train_set_x_orig[index])
#pylab.show()

#打印出当前的训练标签值
#使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
#print("【使用np.squeeze：" + str(np.squeeze(train_set_y[:,index])) + "，不使用np.squeeze： " + str(train_set_y[:,index]) + "】")
#只有压缩后的值才能进行解码操作
print("y=" + str(train_set_y[:,index]) + ", it's a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "' picture")
m_train = train_set_y.shape[1] #训练集里图片的数量。
m_test = test_set_y.shape[1] #测试集里图片的数量。
num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。

#现在看一看我们加载的东西的具体情况
print ("训练集的数量: m_train = " + str(m_train))
print ("测试集的数量 : m_test = " + str(m_test))
print ("每张图片的宽/高 : num_px = " + str(num_px))
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
print ("测试集_标签的维数: " + str(test_set_y.shape))

#X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
#将训练集的维度降低并转置。
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print ("测试集_标签的维数 : " + str(test_set_y.shape))

#使数据位于【0,1】之间，因为图片内容最大不会超过255，所以直接除以255即可。一般在预处理数据时要将数据进行标准化
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    """
        参数：
            z  - 任何大小的标量或numpy数组。

        返回：
            s  -  sigmoid（z）
        """
    s = 1/(1+np.exp(-z))  #注意z前的负号
    return s

#测试sigmoid()
# print("====================测试sigmoid====================")
# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(9.2) = " + str(sigmoid(9.2)))

def initialize_with_zeros(dim):
    """
           此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。

           参数：
               dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）

           返回：
               w  - 维度为（dim，1）的初始化向量。
               b  - 初始化的标量（对应于偏差）
       """
    w = np.zeros(shape=(dim,1))
    b = 0

    #使用断言来判断初始化的是否正确
    assert (w.shape == (dim,1))
    assert (isinstance(b,float) or isinstance(b,int))

    return w,b

def propagate(w,b,X,Y,):
    """
       实现前向和后向传播的成本函数及其梯度。
       参数：
           w  - 权重，大小不等的数组（num_px * num_px * 3，1）
           b  - 偏差，一个标量
           X  - 矩阵类型为（num_px * num_px * 3，训练数量）
           Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

       返回：
           cost- 逻辑回归的负对数似然成本
           dw  - 相对于w的损失梯度，因此与w相同的形状
           db  - 相对于b的损失梯度，因此与b的形状相同
       """
    m = X.shape[1]

    #正向传播
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -1/m *np.sum(Y * np.log(A)+(1-Y) * np.log(1-A))

    #反向传播
    dw = 1/m *np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)

    #使用断言确保我的数据是正确的
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    #使用字典来存储dw db用来梯度下降
    grads ={
        "dw" : dw,
        "db" : db
    }

    return (grads,cost)

#测试一下propagate
# print("====================测试propagate====================")
# #初始化一些参数
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))


def optimize(w,b,X,Y,num_iterations , learning_rate , print_cost = False):
    """
       此函数通过运行梯度下降算法来优化w和b

       参数：
           w  - 权重，大小不等的数组（num_px * num_px * 3，1）
           b  - 偏差，一个标量
           X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
           Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
           num_iterations  - 优化循环的迭代次数
           learning_rate  - 梯度下降更新规则的学习率
           print_cost  - 每100步打印一次损失值

       返回：
           params  - 包含权重w和偏差b的字典
           grads  - 包含权重和偏差相对于成本函数的梯度的字典
           成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。

       提示：
       我们需要写下两个步骤并遍历它们：
           1）计算当前参数的成本和梯度，使用propagate（）。
           2）使用w和b的梯度下降法则更新参数。
       """

    costs = []
    for i in range(num_iterations):

        grads,cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        #每100次迭代就记录cost
        if i%100==0:
            costs.append(cost)
        #每100次迭代打印出误差值惹
        if print_cost and i%100==0:
            print("迭代的次数为 %i,损失为 %f" %(i,cost))

    params={
        "w":w,
        "b":b
    }
    grads ={

        "dw":dw,
        "db":db
    }
    return (params,grads,costs)

# #测试optimize
# print("====================测试optimize====================")
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))


#下面可以使用训练好的w和b来预测数据
def predict(w,b,X):
    """
       使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，

       参数：
           w  - 权重，大小不等的数组（num_px * num_px * 3，1）
           b  - 偏差，一个标量
           X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据

       返回：
           Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）

       """
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0],1)


    #根据sigmoid函数进行预测
    A = sigmoid(np.dot(w.T,X)+b)

    #根据阈值进行分类嗷
    for i in range(X.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0

    assert (Y_prediction.shape == (1,m))
    return Y_prediction

# #测试predict
# print("====================测试predict====================")
# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
# print("predictions = " + str(predict(w, b, X)))

#使用model函数，将上述所有的整合一下，这样在运行时，只需要调用model函数
def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost = False):
    """
       通过调用之前实现的函数来构建逻辑回归模型

       参数：
           X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
           Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
           X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
           Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
           num_iterations  - 表示用于优化参数的迭代次数的超参数
           learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
           print_cost  - 设置为true以每100次迭代打印成本

       返回：
           d  - 包含有关模型信息的字典。
       """
    w,b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w,b = parameters["w"],parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    #打印训练后的准确性
    print("训练集准确性： " ,format(100-np.mean(np.abs(Y_prediction_train - Y_train))*100),"%")
    print("测试集准确性： " ,format(100-np.mean(np.abs(Y_prediction_test - Y_test))*100),"%")

    d = {
        "costs" : costs,
        "Y_prediction_test" : Y_prediction_test,
        "Y_prediction_train" : Y_prediction_train,
        "w" : w,
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations" : num_iterations
    }

    return d


print("====================测试model====================")
#这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#绘制图
# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations(per hundred)')
# plt.title("learning rate = "+ str(d["learning_rate"]))
# plt.show()
    

#通过比较不同的学习率来看一下模型的效果

learning_rates = [0.01,0.001,0.0001]
models = {}
for i in learning_rates:
    print("learning rate is "+str(i))
    models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=1500,learning_rate = i,print_cost=False)
    print("\n"+"---------------------------------"+"\n")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))


plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center',shadow = True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
