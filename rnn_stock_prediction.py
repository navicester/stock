'''
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import os
import sys

# matplotlib.use('TkAgg')

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)                                                         # 分母
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7                                                                                              # 根据前7天的数据
data_dim = 13 # 5                                                                                           # 数据维度 [可修改]
hidden_dim = 10                                                                                             # 隐层
output_dim = 1                                                                                              # 输出层
learning_rate = 0.01                                                                                        # 
iterations = 500                                                                                            #

# Open, High, Low, Volume, Close
xy = np.loadtxt('000016.csv', delimiter=',') # 601328 601288 000016
xy = xy[::-1]  # reverse order (chronically ordered)

# train/test split                                                                                          # 训练集和测试集分割
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]                                                                                # 获取训练集
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence    # 获取测试集

# ls = np.array(test_set)[:-1,-1]#.tolist()
# max_test_set = max(ls) 
# min_test_set = min(ls) 
# print((max_test_set, min_test_set))


# Scale each                                                                                                # 归一化
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):                                                       # 在不同的时间轴上
        _x = time_series[i:i + seq_length, :]                                                               # 前7天数据，包括所有的数据 [行起始:行结束, 列起始：列结束]
        _y = time_series[i + seq_length, [-1]]  # Next close price                                          # 7天之后的这一天，收盘价
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)                                                                 # 返回 x,y

trainX, trainY = build_dataset(train_set, seq_length)                                                       # 生成训练数据集，包括x,y
testX, testY = build_dataset(test_set, seq_length)                                                          # 生成测试数据集，包括x,y


# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])                                                # 第二个参数是矩阵
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network                                                                                      # 循环神经网络(RNN) - LSTM，全称为长短期记忆网络(Long Short Term Memory networks)，是一种特殊的RNN，能够学习到长期依赖关系。
cell = tf.contrib.rnn.BasicLSTMCell(                                                                        # RNN - BasicLSTMCell
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)                                             # tensorflow封装的用来实现递归神经网络（RNN）的函数
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output                        # 添加完全连接的图层

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares                                           # 损失函数
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)                                                           # 寻找全局最优点的优化算法
train = optimizer.minimize(loss)

# RMSE                                                                                                      # 均方根误差（root-mean-square error）
targets = tf.placeholder(tf.float32, [None, 1])                                                             # 实际观察值
predictions = tf.placeholder(tf.float32, [None, 1])                                                         # 预测函数
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))                                            # RMSE公式
# targets_ave = tf.reduce_mean(targets)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={                                                  # 求损失函数和训练模型
                                X: trainX, Y: trainY})
        # print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})                                                   # 求预测值
    rmse_val = sess.run(rmse, feed_dict={                                                                   # 求RMSE
                    targets: testY, predictions: test_predict})

    total_error = tf.reduce_sum(tf.square(tf.subtract (testY, tf.reduce_mean(testY))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract (testY, test_predict)))
    R_squared = 1- unexplained_error/total_error

    print(testY)
    print("RMSE: {}".format(rmse_val))
    print("R2: {}".format(R_squared.eval()))

    # Plot predictions
    plt.plot(testY, color="red", label="close")
    plt.plot(test_predict, color="green", label="predict")
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()