from cgi import test
from tabnanny import verbose
from tokenize import PlainToken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
# 注意力机制
import math
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import Sequential, layers, utils
seed(666)
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   # 用来正常显示负号

# 读取数据集
file_path = r'D:\vscode\机器学习\LSTM\07_基于LSTM单变量预测_电力消耗预测\disp7.csv'
dataset = pd.read_csv(file_path)


# 将字段Datetime数据类型转换为日期类型
dataset['Datetime'] = pd.to_datetime(dataset['Datetime'], format="%Y-%m-%d")

# 将字段Datetime设置为索引列
# 目的：后续基于索引来进行数据集的切分
dataset.index = dataset.Datetime
# print(dataset[-10:])
# 将原始的Datetime字段列删除
dataset.drop(columns=['Datetime'], axis=1, inplace=True)
dataset_or = dataset.copy()
print(dataset_or.head())
# 差分处理
dataset_diff1 = dataset.diff(1)
print(dataset_diff1.head())
dataset_diff1[np.isnan(dataset_diff1)] = 0
dataset_sub = dataset - dataset_diff1
print(dataset_sub.head())

# 数据进行归一化
scaler = MinMaxScaler()
dataset['disp'] = scaler.fit_transform(dataset_diff1['disp'].values.reshape(-1, 1))


# 功能函数：构造特征数据集和标签集
def create_new_dataset(dataset, seq_len = 12):
    '''基于原始数据集构造新的序列特征数据集
    Params:
        dataset : 原始数据集
        seq_len : 序列长度（时间跨度）
    
    Returns:
        X, y
    '''
    X = [] # 初始特征数据集为空列表
    y = [] # 初始标签数据集为空列表
    
    start = 0 # 初始位置
    end = dataset.shape[0] - seq_len # 截止位置
    
    for i in range(start, end): # for循环构造特征数据集
        sample = dataset[i : i+seq_len] # 基于时间跨度seq_len创建样本
        label = dataset[i+seq_len] # 创建sample对应的标签
        X.append(sample) # 保存sample
        y.append(label) # 保存label
    
    # 返回特征数据集和标签集
    return np.array(X), np.array(y)


 #功能函数：基于新的特征的数据集和标签集，切分：X_train, X_test
def split_dataset(X, y, train_ratio=0.8):
    '''基于X和y，切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例
    
    Returns:
        X_train, X_test, y_train, y_test
    '''
    X_len = len(X) # 特征数据集X的样本数量
    train_data_len = int(X_len * train_ratio) # 训练集的样本数量
    
    X_train = X[:train_data_len] # 训练集
    y_train = y[:train_data_len] # 训练标签集
    
    X_test = X[train_data_len:] # 测试集
    y_test = y[train_data_len:] # 测试集标签集
    
    # 返回值
    return X_train, X_test, y_train, y_test



# 功能函数：基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)
def create_batch_data(X, y, batch_size=32, data_type=1):
    '''基于训练集和测试集，创建批数据
    Params:
        X : 特征数据集
        y : 标签数据集
        batch_size : batch的大小，即一个数据块里面有几个样本
        data_type : 数据集类型（测试集表示1，训练集表示2）
   
    Returns:
        train_batch_data 或 test_batch_data
    '''
    if data_type == 1: # 测试集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) # 封装X和y，成为tensor类型
        test_batch_data = dataset.batch(batch_size) # 构造批数据
        # 返回
        return test_batch_data
    else: # 训练集
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) # 封装X和y，成为tensor类型
        train_batch_data = dataset.cache().shuffle(1000).batch(batch_size) # 构造批数据
        # 返回
        return train_batch_data


# ① 原始数据集
dataset_original = dataset


# ② 构造特征数据集和标签集，seq_len序列长度为12小时
SEQ_LEN = 6 # 序列长度
X, y = create_new_dataset(dataset_original.values, seq_len = SEQ_LEN)


# ③ 数据集切分
X_train, X_test, y_train, y_test = split_dataset(X, y, train_ratio=0.8)



# ④ 基于新的X_train, X_test, y_train, y_test创建批数据(batch dataset)
# 测试批数据
test_batch_dataset = create_batch_data(X_test, y_test, batch_size=32, data_type=1)


# 训练批数据
train_batch_dataset = create_batch_data(X_train, y_train, batch_size=32, data_type=2)


# model = Sequential([
#     layers.LSTM(4, input_shape=(SEQ_LEN, 1),  return_sequences=True),
#     layers.LSTM(2),
#     layers.Dense(1)
# ])
# model.compile(optimizer='adam', loss="mae")

def eca_block(inputs, b=1, gamma=2):
    channel = inputs.shape[-2]
    #print(channel)
    kernel_size = int(abs(math.log(512,2) +b) / gamma)
    kernel_size = kernel_size if kernel_size % 2 else kernel_size +1
    # c
    x = layers.GlobalAveragePooling2D()(inputs)
    #print(x.shape)
    # c,1
    rx = layers.Reshape([-1, 1])(x)
    # c,1
    cx = layers.Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False,)(rx)
    ax = layers.Activation('sigmoid')(cx)
    # 1,1,c
    rx2 = layers.Reshape([1,1,-1])(ax)

    return layers.Multiply()([inputs, rx2])

class AE_LSTM(keras.Model):
    def __init__(self, AL=0):
        super(AE_LSTM, self).__init__()
        self.AL = AL
        #self.lstm1 = layers.LSTM(512, input_shape=(SEQ_LEN, 1), return_sequences=True) #r2: 0.09741314775888865 RMSE: 0.5416388667721437 MAE: 0.365136974400946
        self.lstm1 = layers.GRU(256, input_shape=(SEQ_LEN, 1), return_sequences=True)  #r2: 0.2396863009387643 RMSE: 0.4971205621925335 MAE: 0.3718372108112253
        self.drop1 = layers.Dropout(0.5)
        #self.lstm2 = layers.LSTM(units=4)
        self.lstm2 = layers.GRU(4)
        self.drop2 = layers.Dropout(0.3)
        self.fc1 = layers.Dense(1)

    def call(self, inputs, training=None):
        # x [b,24,4]
        x = inputs
        print(x.shape)
        if self.AL == 1:
            #print(x.shape)
            # x = tf.expand_dims(x, axis=1)
            # # print('ex', x.shape)
            # x = eca_block(x)   #r2: r2: -4.685496463252987 RMSE: 1.5428981000561628 MAE: 1.5130880361665036
            #x = tf.reshape(x,[-1,SEQ_LEN,1])
            x = attention_3d_block2(x)
            # print(x.shape)
        x = self.lstm1(x, training=training)  # r2: -5.271222786849109 RMSE: 1.6204261051399234 MAE: 1.592025185214355
        x = self.drop1(x)
        #x = attention_3d_block2(x)
        x = self.lstm2(x, training=training)
        x = self.drop2(x)
        out = self.fc1(x)
        return out

def attention_3d_block2(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = layers.Permute((2, 1))(inputs)  #Permute(dims)置换输入的维度。
    a = layers.Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = layers.Lambda(lambda x: K.mean(x, axis=1))(a)
        a = layers.RepeatVector(input_dim)(a)

    a_probs = layers.Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul

# 定义 checkpoint，保存权重文件

file_path = "best_checkpoint.hdf5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                         monitor='loss', 
                                                         mode='min', 
                                                         save_best_only=True,
                                                         save_weights_only=True)


# 模型编译
model = AE_LSTM(0)   # 1 为加入注意力机制  0为不加（默认值）
model.compile(optimizer = keras.optimizers.Adam(0.0001), loss="mae")



# 模型训练

history = model.fit(train_batch_dataset,
          epochs=1000,
          validation_data=test_batch_dataset,
          callbacks=[checkpoint_callback])


# 显示 train loss 和 val loss
plt.figure(figsize=(16,8))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("LOSS")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

# 模型验证
test_pred = model.predict(X_test, verbose=1)  # verbose:1代表显示进度条
fit_pred = model.predict(X_train, verbose=1)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



# 反归一化
y_test_new = scaler.inverse_transform(y_test)
test_pred_new = scaler.inverse_transform(test_pred)
fit_data = scaler.inverse_transform(fit_pred)

# print(y_test_new.shape)
# 加上差分值

# y_test_new = y_test_new.reshape(-1)
test_labels_new = y_test_new + dataset_sub['disp'].values[-len(test_pred_new):].reshape(-1,1)
# test_pred_new = test_pred_new.reshape(-1)
# print(test_pred_new)
test_predict_new = test_pred_new + dataset_sub['disp'].values[-len(test_pred_new):].reshape(-1,1)
# print(test_predict_new)
train_true = dataset_or['disp'].values[:-len(test_pred_new)+1].reshape(-1,1)
#print(fit_data.shape,dataset_sub.shape)
fit_data = fit_data + dataset_sub['disp'].values[SEQ_LEN+1:-len(test_pred_new)+1].reshape(-1,1)
#print(dataset_sub['disp'].values)

# 计算精度评价指标r2
score = r2_score(test_labels_new, test_predict_new)
rmse = np.sqrt(mean_squared_error(test_labels_new, test_predict_new))
mae = mean_absolute_error(test_labels_new, test_predict_new)

print('r2:',score,'RMSE:',rmse,'MAE:',mae)
# # 绘制模型验证结果

# plt.figure(figsize=(16,8))
# plt.plot(y_test_new, label="True label")
# plt.plot(test_pred_new, label="Pred label")
# plt.title("True vs Pred")
# plt.legend(loc='best')
# plt.show()


# 绘制test中前100个点的真值与预测值
# y_true = y_test_new
# y_pred = test_pred_new
# true_data = train_true

# 处理为DataFrame格式
fit_data = pd.DataFrame(fit_data, index=dataset_or[SEQ_LEN:-len(test_pred_new)].index, columns=['fit'])
y_true = pd.DataFrame(test_labels_new, index=dataset_or[-len(test_pred_new):].index, columns=['true'])
y_pred = pd.DataFrame(test_predict_new, index=dataset_or[-len(test_pred_new):].index, columns=['true'])
# fig, axes = plt.subplots(2, 1, figsize=(16,8))
# axes[0].plot(y_true, marker='o', color='red')
# axes[1].plot(y_pred, marker='*', color='blue')
plt.figure(figsize=(16,8))
plt.plot(y_true, marker='o', color='red', label="True data")
plt.plot(y_pred, marker='*', color='blue', label="Pred data")
plt.plot(dataset_or[SEQ_LEN:-len(test_pred_new)], marker='*', color='green', label="Train data")
plt.plot(fit_data, marker='o', color='gray', label="Fit data")
# plt.plot(dataset_or, marker='*', color='black', label="Fit data")
plt.title('RMSE : %.4f   MAE : %.4f' % (rmse, mae))
plt.xlabel('日期', fontsize=12, verticalalignment='top')
plt.ylabel('形变量/mm', fontsize=14, horizontalalignment='center')
plt.grid()  # 生成网格
plt.legend(loc='best')
plt.show()


# 模型预测未来数据
# 选择test中的最后一个样本
sample = X_test[-1]
sample = sample.reshape(1, sample.shape[0], 1)

# 模型预测
sample_pred = model.predict(sample)
# 预测后续20个数据点
ture_data = X_test[-1] # 真实test的最后20个数据点

def predict_next(model, sample, epoch=20):
    temp1 = list(sample[:,0])
    for i in range(epoch):
        sample = sample.reshape(1, SEQ_LEN, 1)
        pred = model.predict(sample)
        value = pred.tolist()[0][0]
        temp1.append(value)
        sample = np.array(temp1[i+1 : i+SEQ_LEN+1])
    return np.array(temp1)

preds = predict_next(model, ture_data, 20)
# 反归一化
pred_new = scaler.inverse_transform(np.array(preds).reshape(-1,1))
print(pred_new)
plt.figure(figsize=(12,6))
plt.plot(preds, color='yellow', label='Prediction')
plt.plot(ture_data, color='blue', label='Truth')
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend(loc='best')
plt.show()
