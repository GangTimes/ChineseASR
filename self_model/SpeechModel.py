#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import platform as plat
import os
import time
from tqdm import tqdm
import keras as kr
import numpy as np
import random

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization # , Flatten
from keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge
from keras import backend as K
from keras.optimizers import SGD, Adadelta, Adam
from Data import DataSpeech,ConfigSpeech
from Data import ConfigSpeech as config
from Utils import get_edit_distance
class ModelSpeech(DataSpeech): # 语音模型类
    def __init__(self):
        super(ModelSpeech,self).__init__()
        self.create_model()
        self.create_ctc()
        self.create_opt()
    def create_model(self):
        '''
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

        '''

        self.input_data = Input(name='the_inputs', shape=(None, self.audio_feature_len, 1))
        layer_h1 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(self.input_data) # 卷积层
        layer_h1=BatchNormalization()(layer_h1)
        layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1) # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2) # 池化层
        layer_h3=BatchNormalization()(layer_h3)
        layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3) # 卷积层
        layer_h4=BatchNormalization()(layer_h4)
        layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4) # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
        layer_h6=BatchNormalization()(layer_h6)
        layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6) # 卷积层
        layer_h7=BatchNormalization()(layer_h7)
        layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7) # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8) # 池化层
        layer_h9=BatchNormalization()(layer_h9)
        layer_h10 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9) # 卷积层
        layer_h10=BatchNormalization()(layer_h10)
        layer_h11 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10) # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11) # 池化层
        layer_h12=BatchNormalization()(layer_h12)
        layer_h13 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h12) # 卷积层
        layer_h13=BatchNormalization()(layer_h13)
        layer_h14 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h13) # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14) # 池化层
        layer_h16 = Reshape((-1, 3200))(layer_h15) #Reshape层
        layer_h16 = Dropout(0.3)(layer_h16)
        layer_h16=BatchNormalization()(layer_h16)
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16) # 全连接层
        layer_h17=BatchNormalization()(layer_h17)
        layer_h18 = Dense(self.output_size, use_bias=True, kernel_initializer='he_normal')(layer_h17) # 全连接层
        self.y_pred = Activation('softmax', name='Activation0')(layer_h18)
        self.predict_model= Model(inputs=self.input_data, outputs=self.y_pred)
    def create_ctc(self):
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([self.y_pred, self.labels, self.input_length, self.label_length])
        self.ctc_model = Model(inputs=[self.input_data, self.labels, self.input_length, self.label_length], outputs=self.loss_out)
        self.ctc_model.summary()
    def create_metric(self):
        pass
    def create_opt(self):
        opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
        self.ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)


    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    def decode_ctc(self,result):
        result=result[:,:,:]
        in_len=np.zeros((1),dtype=np.int32)
        in_len[0]=result.shape[1]
        r=K.ctc_decode(result,in_len,greedy=True,beam_width=1,top_paths=1)
        rv=K.get_value(r[0][0])[0]
        text=[self.id2py[i] for i in rv]
        return rv,text
def evaluate(model=None,data=None):
    if data==None:
        data=DataSpeech()
    if model==None:
        model=ModelSpeech()
    data_iters=data.create_batch('dev',False)
    word_total=0
    word_error_num=0
    for i in tqdm(range(data.batch_num['dev'])):
        batch=next(data_iters)
        results=model.predict_model.predict_on_batch(batch[0]['the_inputs'])
        for result,label in zip(results,batch[0]['the_labels']):
            result=result.reshape((1,result.shape[0],result.shape[1]))
            pre,text=model.decode_ctc(result)
            word_num=len(label)
            word_total+=word_num
            edit_distance=get_edit_distance(label,pre)
            if edit_distance<=word_num:
                word_error_num+=edit_distance
            else:
                word_error_num+=word_num
    print('语音识别错误率:{.4f}'+word_error_num/word_total)


def train(model=None,data=None):
    if data==None:
        data=DataSpeech()
    if model==None:
        model=ModelSpeech()
    for epoch in range(1,config.epochs+1):
        data_iters=data.create_batch('dev')
        dev_iters=data.create_batch('test',False)
        model.ctc_model.fit_generator(data_iters,steps_per_epoch=data.batch_num['dev'],epochs=1,validation_data=dev_iters,validation_steps=data.batch_num['test'])
        model.ctc_model.save_weights(config.model_path)
        evaluate(model,data)

def test(model=None,data=None):
    if data==None:
        data=DataSpeech()
    if model==None:
        model=ModelSpeech()
    data_iters=data.create_batch('test',False)
    for batch in data_iters:
        results=model.predict_model.predict_on_batch(batch[0]['the_inputs'])

        for result in results:
            result=result.reshape((1,result.shape[0],result.shape[1]))
            pre,text=model.decode_ctc(result)

def main():
    model=ModelSpeech()
    data=DataSpeech()
    if os.path.exists(config.model_path):
        model.ctc_model.load_weights(config.model_path)
    train(model,data)
if __name__=="__main__":
    main()
