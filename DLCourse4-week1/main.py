import numpy as np
import h5py
import matplotlib.pyplot as plt

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# #ipython很好用，但是如果在ipython里已经import过的模块修改后需要重新reload就需要这样
# #在执行用户代码前，重新装入软件的扩展和模块。
# %load_ext autoreload
# #autoreload 2：装入所有 %aimport 不包含的模块。
# %autoreload 2

np.random.seed(1)      #指定随机种子

#使用np.pad()进行填充
def zero_pad(X,pad):
    X_paded = np.pad(X,(
                    (0,0),  #第一维样本数，不填充，第二维图像高度，第三维图像宽度，第四维，通道数，不填充
                    (pad,pad),
                    (pad,pad),
                    (0,0)),
                     'constant',constant_values=0)
    return  X_paded

# np.random.seed(1)
# x = np.random.randn(4,3,3,2)
# x_paded = zero_pad(x,2)
# #查看信息
# print ("x.shape =", x.shape)
# print ("x_paded.shape =", x_paded.shape)
# print ("x[1, 1] =", x[1, 1])
# print ("x_paded[1, 1] =", x_paded[1, 1])
#
# #绘制图
# fig , axarr = plt.subplots(1,2)  #一行两列
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0,:,:,0])
# plt.show()

#单步卷积的实现
def conv_single_step(a_slice_prev,W,b):
    s = np.multiply(a_slice_prev,W) +b
    Z = np.sum(s)
    return Z

np.random.seed(1)

#这里切片大小和过滤器大小相同
# a_slice_prev = np.random.randn(4,4,3)
# W = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
#
# Z = conv_single_step(a_slice_prev,W,b)
#
# print("Z = " + str(Z))


def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积函数的前向传播

    参数：
        A_prev - 上一层的激活输出矩阵，维度为(m, n_H_prev, n_W_prev, n_C_prev)，（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
        W - 权重矩阵，维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）
        b - 偏置矩阵，维度为(1, 1, 1, n_C)，（1,1,1,这一层的过滤器数量）
        hparameters - 包含了"stride"与 "pad"的超参数字典。

    返回：
        Z - 卷积输出，维度为(m, n_H, n_W, n_C)，（样本数，图像的高度，图像的宽度，过滤器数量）
        cache - 缓存了一些反向传播函数conv_backward()需要的一些数据
    """

    #获取来自上一层数据的基本信息
    (m , n_H_prev , n_W_prev , n_C_prev) = A_prev.shape

    #获取权重矩阵的基本信息
    ( f , f ,n_C_prev , n_C ) = W.shape

    #获取超参数hparameters的值
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    #计算卷积后的图像的宽度高度，参考上面的公式，使用int()来进行板除
    n_H = int(( n_H_prev - f + 2 * pad )/ stride) + 1
    n_W = int(( n_W_prev - f + 2 * pad )/ stride) + 1

    #使用0来初始化卷积输出Z
    Z = np.zeros((m,n_H,n_W,n_C))

    #通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):                              #遍历样本
        a_prev_pad = A_prev_pad[i]                  #选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):                        #在输出的垂直轴上循环
            for w in range(n_W):                    #在输出的水平轴上循环
                for c in range(n_C):                #循环遍历输出的通道
                    #定位当前的切片位置
                    vert_start = h * stride         #竖向，开始的位置
                    vert_end = vert_start + f       #竖向，结束的位置
                    horiz_start = w * stride        #横向，开始的位置
                    horiz_end = horiz_start + f     #横向，结束的位置
                    #切片位置定位好了我们就把它取出来,需要注意的是我们是“穿透”取出来的，
                    #自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    #执行单步卷积
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[: ,: ,: ,c],b[0,0,0,c])

    #数据处理完毕，验证数据格式是否正确
    assert(Z.shape == (m , n_H , n_W , n_C ))

    #存储一些缓存值，以便于反向传播使用
    cache = (A_prev,W,b,hparameters)

    return (Z , cache)



# np.random.seed(1)
#
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
#
# hparameters = {"pad" : 2, "stride": 1}
#
# Z , cache_conv = conv_forward(A_prev,W,b,hparameters)
#
# print("np.mean(Z) = ", np.mean(Z))
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

def pool_forward(A_prev,hparameters,mode="max"):



    (m , n_H_prev , n_W_prev , n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev - f) / stride ) + 1
    n_W = int((n_W_prev - f) / stride ) + 1
    n_C = n_C_prev


    A = np.zeros((m , n_H , n_W , n_C))

    for i in range(m):                              #遍历样本
        for h in range(n_H):                        #在输出的垂直轴上循环
            for w in range(n_W):                    #在输出的水平轴上循环
                for c in range(n_C):                #循环遍历输出的通道
                    #定位当前的切片位置
                    vert_start = h * stride         #竖向，开始的位置
                    vert_end = vert_start + f       #竖向，结束的位置
                    horiz_start = w * stride        #横向，开始的位置
                    horiz_end = horiz_start + f     #横向，结束的位置
                    #定位完毕，开始切割
                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]


                    if mode == "max":
                        A[ i , h , w , c ] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[ i , h , w , c ] = np.mean(a_slice_prev)




    cache = (A_prev,hparameters)

    return A,cache

# np.random.seed(1)
# A_prev = np.random.randn(2,4,4,3)
# hparameters = {"f":4 , "stride":1}
#
# A , cache = pool_forward(A_prev,hparameters,mode="max")
# A, cache = pool_forward(A_prev, hparameters)
# print("mode = max")
# print("A =", A)
# print("----------------------------")
# A, cache = pool_forward(A_prev, hparameters, mode = "average")
# print("mode = average")
# print("A =", A)

def conv_backward(dZ,cache):
    (A_prev,W,b,hparameters) = cache
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    (m,n_H,n_W,n_C)  = dZ.shape

    (f,f,n_C_prev,n_C) = W.shape

    pad = hparameters['pad']
    stride = hparameters['stride']

    #intialize the grad
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))

    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start+f
                    horiz_start = w
                    horiz_end = horiz_start+f

                    #切片
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] +=W[:,:,:,c] * dZ[i,h,w,c]

                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]

                    db[:,:,:,c] += dZ[i,h,w,c]


    #取出非填充的数据
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad,pad:-pad,:]

    return (dA_prev,dW,db)

# np.random.seed(1)
# #初始化参数
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 2, "stride": 1}
#
# #前向传播
# Z , cache_conv = conv_forward(A_prev,W,b,hparameters)
# #反向传播
# dA , dW , db = conv_backward(Z,cache_conv)
# print("dA_mean =", np.mean(dA))
# print("dW_mean =", np.mean(dW))
# print("db_mean =", np.mean(db))

#池化层的反向传播：分为max_pooling和average,
#创建掩码矩阵，来记录最大值存在的地方~
def create_mask_from_window(x):
    mask = x==np.max(x)
    return mask

# np.random.seed(1)
#
# x = np.random.randn(2,3)
#
# mask = create_mask_from_window(x)
#
# print("x = " + str(x))
# print("mask = " + str(mask))
def distribute_value(dz,shape):
    (n_H,n_W) = shape

    average = dz/(n_W*n_H)

    a = np.ones(shape) *average

    return a
# dz = 2
# shape = (2,2)
#
# a = distribute_value(dz,shape)
# print("a = " + str(a))


#池化层的反向传播，默认为max
def pool_backward(dA,cache,mode = 'max'):

    (A_prev,hparameters) = cache

    f = hparameters['f']
    stride = hparameters['stride']

    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (m,n_H,n_W,n_C) = dA.shape


    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start+f
                    horiz_start = w
                    horiz_end = horiz_start+f

                    if mode =='max':
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += np.multiply(mask,dA[i,h,w,c])

                    elif mode == 'average':
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += distribute_value(da,shape)
    return dA_prev

np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])




