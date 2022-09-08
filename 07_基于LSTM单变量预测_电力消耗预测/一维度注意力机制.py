from cProfile import label
from multiprocessing.sharedctypes import Value
from pyexpat import model
from statistics import mode
# from tkinter.ttk import _Padding
from unicodedata import name
from numpy.random import seed
import pandas as pd
seed(1)

from keras import backend as k   #导入后端模块
from keras.engine.topology import Layer


class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs) -> None:
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(3, input_shape[2], self.output_dim),
        initializer='uniform',
        trainable=True)
        super(Self_Attention, self).build(input_shape)
    
    def call(self, x):
        WQ = k.dot(x, self.kernel[0])
        Wk = k.dot(x, self.kernel[0])
        WV = k.dot(x, self.kernel[0])

        QK = k.batch_dot(WQ, k.permute_dimensions(Wk, [0,2,1]))
        QK = QK / (64 ** 0.5)
        QK = k.softmax(QK)
        V = k.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim

# 读取数据
file_path = r'D:\vscode\机器学习\LSTM\07_基于LSTM单变量预测_电力消耗预测\disp.csv'
dataset = pd.read_csv(file_path)
values = dataset.values
# 将字段Datetime数据类型转换为日期类型
# dataset['Datetime'] = pd.to_datetime(dataset['Datetime'], format="%Y-%m-%d")
# values = dataset['disp'].values.reshape(-1,1)
values[:,0] = values[:,0]/1000
print(values)

# 归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
XY = scaler.fit_transform(values)
X = XY[:,0]
Y = XY[:,1]

# 划分数据集合
n_train_hours = 80
trainX = X[1:n_train_hours]
trainY = Y[1:n_train_hours]
testX = X[n_train_hours+1:]
testY = Y[n_train_hours+1:]

train3DX = trainX.reshape(trainX.shape[0], -1, 1)
test3DX = trainX.reshape(testX.shape[0], -1, 1)

# 模型搭建
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten

model = Sequential()
model.add(Conv1D(filters=80, kernel_size=1), padding='same', strides=1, activation='relu',
        input_shape=(train3DX.shape[1], train3DX.shape[2]))
model.add(MaxPooling1D(pool_size=1))
model.add(Self_Attention(3))
model.add(Flatten())
model.add(Dense(nuits=1000, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# 编译模型
model.compile(loss='mae', optimizer='adam')

# 训练模型
history = model.fit(train3DX, trainY, batch_size=3, validation_data=(test3DX, testY),
        verbose=2, shuffle=False)

# 评估模型
import matplotlib.pyplot as plt
plt.plot(history.history['loss'],  label='train')
plt.plot(history.history['loss'],  label='test')
plt.legend()
plt.show()

# 预测与反归一化
import pandas as np
forecasttestY0 = model.predict(test3DX)
inv_yhat = np.concatenate((testX, forecasttestY0), axis=1)
inv_y = scaler.inverse_transform(inv_yhat)
forecasttestY = inv_y[:,1]

#评价模型 rmse/mae
