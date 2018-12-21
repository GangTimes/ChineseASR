# -----------------------------------------------------------------------------------------------------
'''
&usage:     数据读取文件
@author:    hongwen sun
'''
# -----------------------------------------------------------------------------------------------------
import os
import random
import sys
import numpy as np
import scipy.io.wavfile as wav
from keras.preprocessing.sequence import pad_sequences
from extra_utils.feature_extract import compute_fbank


class get_data():
    """获取四个数据集音频数据和标注数据"""
    def __init__(self, datapath = '/data/dataset/', read_type = 'train', batch_size = 4):
        super(get_data, self).__init__()
        self.datapath = datapath
        self.wavpath_total = '/data/dataset/'
        self.read_type = read_type
        self.batch_size = batch_size
        # 四个数据集文件放置位置 [data/thchs30]
        self.thchs30_dir = datapath + 'thchs30/'
        self.aishell_dir = datapath + 'aishell/'
        self.primewords_dir = datapath +'primewords/'
        self.stcmds_dir = datapath + 'st-cmds/'
        # 定义标注拼音映射为数字的字典 ['a1', 'a2', ...]
        self.py2id,self.id2py= self.gen_label_dict()
        self.dict_len = len(self.py2id)
        # 四个数据集标签的字典 [label id : label content]
        self.thchs30_label_dict, self.thchs30_label_ids = self.read_label(self.thchs30_dir)
        self.aishell_label_dict, self.aishell_label_ids = self.read_label(self.aishell_dir)
        self.primewords_label_dict, self.primewords_label_ids = self.read_label(self.primewords_dir)
        self.stcmds_label_dict, self.stcmds_label_ids = self.read_label(self.stcmds_dir)
        # 四个数据集音频lsit的字典 [wav id : wav path]
        self.thchs30_wav_paths = self.read_wavpath(self.thchs30_dir)
        self.aishell_wav_paths = self.read_wavpath(self.aishell_dir)
        self.primewords_wav_paths = self.read_wavpath(self.primewords_dir)
        self.stcmds_wav_paths = self.read_wavpath(self.stcmds_dir)
        # 四个数据集语料的个数
        self.total_data_num = self.get_data_num()


    # label到数字的映射,最好不要每次生成，尽量直接使用一个完整的拼音映射，这里就是固定死了
    def gen_label_dict(self):
        py2id={}
        id2py={}
        with open('/data/dataset/dict/py2id_dict.txt','r') as file:
            for line in file:
                py,idx=line.strip('\n').strip().split('\t')
                py2id[py]=int(idx)
                id2py[int(idx)]=py
        return py2id,id2py


    # 读取标签，返回标签文件到对应字典
    def read_label(self, dirpath):
        # dirpath = 'data/thchs30/' | read_type = 'train' | label_path = 'data/thchs30/train.syllabel.txt'
        label_path = dirpath + self.read_type + '.syllabel.txt'
        label_cont = open(label_path, 'r', encoding='utf-8')
        label_dict = {}
        label_ids = []
        for content in label_cont:
            content = content.strip('\n')
            cont_id = content.split(' ', 1)[0]
            cont_lb = content.split(' ', 1)[1]
            cont_nu = []
            # 通过字典将拼音标注转化为数字标注
            for label in cont_lb.split(' '):
                try:
                    cont_nu.append(self.py2id[label])
                except:
                    pass
            #cont_nu = [self.label_dict.index(label) for label in cont_lb.split(' ')]
            # 我们定义标签为50个值，不够补零, 0对应的是未知的拼音
            #add_num = list(np.zeros(50-len(cont_nu)))
            # dict['A11_180'] = 'liu2 xiao3 guang1 jiu3 ...'
            label_dict[cont_id] = cont_nu
            label_ids.append(cont_id)
        return label_dict, label_ids


    # 读取音频地址
    def read_wavpath(self, dirpath):
        # dirpath = 'data/thchs30/' | read_type = 'train' | label_path = 'data/thchs30/train.wav.lst'
        wavfiles_path = dirpath + self.read_type + '.wav.lst'
        wavfiles_list = open(wavfiles_path, 'r')
        wavfiles_dict = {}
        for wavpath in wavfiles_list:
            wavpath = wavpath.strip('\n')
            wav_id = wavpath.split(' ', 1)[0]
            wav_path = wavpath.split(' ', 1)[1]
            wavfiles_dict[wav_id] = self.wavpath_total + wav_path
        return wavfiles_dict

    # 获取四个数据集的总量
    def get_data_num(self):
        total_data_num = len(self.thchs30_wav_paths) + len(self.aishell_wav_paths) \
                        + len(self.primewords_wav_paths) + len(self.stcmds_wav_paths)
        return total_data_num

    # 用于训练或者测试的数据生成器
    def data_generator(self):
        thchs30_num = len(self.thchs30_wav_paths)
        stcmds_num = len(self.stcmds_wav_paths)
        aishell_num = len(self.aishell_wav_paths)
        primewd_num = len(self.primewords_wav_paths)
        loop = -1
        while True:
            loop = loop + 1
            feats = []
            labels = []
            input_lengths = []
            label_lengths = []
            for i in range(self.batch_size):
                # 获取随机获得的一个数字，由这个随机数字获得一个id，通过id选择数据
                if self.read_type == 'train':
                    wav_choice = random.randint(0, self.total_data_num - 1)
                else:   # 测试模式下不需要打乱数据顺序
                    wav_choice = loop * self.batch_size + i
                #print('wav choice is:', wav_choice)
                if wav_choice < thchs30_num:
                    wavpath = self.thchs30_wav_paths[self.thchs30_label_ids[wav_choice]]
                    label = self.thchs30_label_dict[self.thchs30_label_ids[wav_choice]]
                elif wav_choice < (thchs30_num + stcmds_num):
                    wavpath = self.stcmds_wav_paths[self.stcmds_label_ids[wav_choice - thchs30_num]]
                    label = self.stcmds_label_dict[self.stcmds_label_ids[wav_choice - thchs30_num]]
                elif wav_choice < (thchs30_num + stcmds_num + aishell_num):
                    wavpath = self.aishell_wav_paths[self.aishell_label_ids[wav_choice - thchs30_num - stcmds_num]]
                    label = self.aishell_label_dict[self.aishell_label_ids[wav_choice - thchs30_num - stcmds_num]]
                else:
                    wavpath = self.primewords_wav_paths[self.primewords_label_ids[wav_choice - thchs30_num - stcmds_num - aishell_num]]
                    label = self.primewords_label_dict[self.primewords_label_ids[wav_choice - thchs30_num - stcmds_num - aishell_num]]
                # wavpath是一个音频文件路径， label是数字化后长度为50的array
                # 提取特征，生成数据
                fbank_feat = compute_fbank(wavpath)
                # 如果有超过2000的数据，则不取
                if fbank_feat.shape[1] > 2000:
                    continue
                input_length = fbank_feat.shape[1]//8 + 1
                fbank_feat = pad_sequences(fbank_feat, maxlen=2000, dtype='float', padding='post', truncating='post').T
                fbank_feat = fbank_feat.reshape(fbank_feat.shape[0], fbank_feat.shape[1], 1)
                input_lengths.append(input_length)
                feats.append(fbank_feat)
                label_length = len(label)
                label_lengths.append(label_length)
                add_num = list(np.zeros(50-label_length))
                label = np.array(label + add_num)
                labels.append(label)
            label_lengths = np.array(label_lengths)
            input_lengths = np.array(input_lengths)
            feats = np.array(feats)
            labels = np.array(labels)
            # 调用get_batch将数据处理为训练所需的格式
            inputs, outputs = self.get_batch(feats, labels, label_length=label_lengths, input_length=input_lengths)
            yield inputs, outputs


    # 将数据格式整理为能够被网络所接受的格式，被data_generator调用
    def get_batch(self, x, y, train=False, label_length=30, input_length=90):
        X = np.expand_dims(x, axis=4)
        X = x # for model2
    #     labels = np.ones((y.shape[0], max_pred_len)) *  -1 # 3 # , dtype=np.uint8
        labels = y
        #input_length = np.ones([x.shape[0], 1]) * ( input_length)
        input_length = input_length
        #print(input_length)
        #print(label_length)
        label_length = label_length
    #     label_length = np.ones([y.shape[0], 1])
        #label_length = np.sum(labels > 0, axis=1)
        #label_length = np.expand_dims(label_length,1)
        inputs = {'the_input': X,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([x.shape[0]])}  # dummy data for dummy loss function
        return (inputs, outputs)



    def label_generator(self):
        thchs30_num = len(self.thchs30_wav_paths)
        stcmds_num = len(self.stcmds_wav_paths)
        aishell_num = len(self.aishell_wav_paths)
        primewd_num = len(self.primewords_wav_paths)
        loop = 39
        while True:
            loop = loop + 1
            for i in range(self.batch_size):
                # 获取随机获得的一个数字，由这个随机数字获得一个id，通过id选择数据
                if self.read_type == 'train':
                    wav_choice = random.randint(0, self.total_data_num - 1)
                else:   # 测试模式下不需要打乱数据顺序
                    wav_choice = loop * self.batch_size + i
                #print('wav choice is:', wav_choice)
                if wav_choice < thchs30_num:
                    label = self.thchs30_label_dict[self.thchs30_label_ids[wav_choice]]
                elif wav_choice < (thchs30_num + stcmds_num):
                    label = self.stcmds_label_dict[self.stcmds_label_ids[wav_choice - thchs30_num]]
                elif wav_choice < (thchs30_num + stcmds_num + aishell_num):
                    label = self.aishell_label_dict[self.aishell_label_ids[wav_choice - thchs30_num - stcmds_num]]
                else:
                    label = self.primewords_label_dict[self.primewords_label_ids[wav_choice - thchs30_num - stcmds_num - aishell_num]]

            yield label




