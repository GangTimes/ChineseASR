#!/usr/bin/env python
# coding=utf-8

import os
import random
import numpy as np
from Utils import compute_fbank
from keras.preprocessing.sequence import pad_sequences
class DataConfig():
    base_dir='/data/dataset/'
    data_names=['st-cmds','thchs30','primewords','aishell']
    data_dirs={name:'/data/dataset/'+name+'/' for name in data_names}
    wav2py_paths={}
    types=['train','test','dev']
    for type in types:
        temp={}
        for name in ['syllabel','wav']:
            temp[name]=type+'.'+name+'.txt' if name!='wav' else type+'.'+name+'.lst'
        wav2py_paths[type]=temp
    dict_dir=base_dir+'dict/'
    py2id_dict=dict_dir+'py2id_dict.txt'
    hz2id_dict=dict_dir+'hz2id_dict.txt'
    py2hz_dict=dict_dir+'py2hz_dict.txt'
    py2hz_dir=base_dir+'pinyin2hanzi/'

class ConfigSpeech(DataConfig):
    output_size=1472
    label_len=50
    audio_len=2000
    audio_feature_len=200
    epochs=10
    save_step=1000
    batch_size=16
    dev_num=10
    train_num=100
    model_dir='models/speech_model/fix/'
    model_name='speechfull.model'
    model_path=model_dir+model_name
    log_dir='log/'
    log_name='speechfixfull.txt'
    log_path=log_dir+log_name

class DataSpeech(ConfigSpeech):
    def __init__(self):
        super(DataSpeech,self).__init__()
        self.create_dict()
        self.create_wav2py()
    def create_wav2py(self):
        self.wav2py={}
        self.batch_num={}
        for _type,path in self.wav2py_paths.items():
            self.wav2py[_type]={}
            start_num=0
            for name,data_dir in self.data_dirs.items():
                id2wav={}
                id2py={}
                with open(data_dir+self.wav2py_paths[_type]['wav'],'r',encoding='utf-8') as file:
                    for line in file:
                        idx,path=line.strip('\n').strip().split(' ')
                        id2wav[idx.strip()]=self.base_dir+path.strip()
                with open(data_dir+self.wav2py_paths[_type]['syllabel'],'r',encoding='utf-8') as file:
                    for line in file:
                        ws=line.strip('\n').strip().split(' ')
                        idx,pys=ws[0],ws[1:]
                        id2py[idx.strip()]=pys
                assert len(id2py)==len(id2wav)
                for idx,key in enumerate(id2py.keys()):
                    self.wav2py[_type][start_num+idx]=(id2wav[key],id2py[key])
                start_num=len(self.wav2py[_type])
            batch_num=start_num //self.batch_size
            self.batch_num[_type]=batch_num if start_num%self.batch_size==0 else batch_num+1

    def create_batch(self,flag='train',shuffle=True):
        data_num=len(self.wav2py[flag])
        idxs=list(range(data_num))
        if shuffle:
            random.shuffle(idxs)
        wavs=[]
        labels=[]
        label_lengths=[]
        input_lengths=[]
        for i,idx in enumerate(idxs):
            wav_path,pys=self.wav2py[flag][idx]
            fbank=compute_fbank(wav_path)
            label=[self.py2id[py] for py in pys]
            while((fbank.shape[1]>=self.audio_len ) or (len(label)>=self.label_len)):
                temp=random.randint(len(idxs)//4,len(idxs)//2)
                wav_path,pys=self.wav2py[flag][temp]
                fbank=compute_fbank(wav_path)
                label=[self.py2id[py] for py in pys]

            assert len(wavs)==len(labels)
            if len(wavs)==self.batch_size:
                wavs=np.array(wavs)
                the_inputs=np.expand_dims(wavs,axis=4)
                the_inputs=wavs.transpose((0,2,1,3))
                the_labels=np.array(labels)
                input_lengths=np.array(input_lengths)
                label_lengths=np.array(label_lengths)
                inputs=[the_inputs,the_labels,input_lengths,label_lengths]
                outputs=np.zeros([self.batch_size,1],dtype=np.float32)
                yield inputs,outputs
                wavs,labels,label_lengths,input_lengths=[],[],[],[]
            input_length=fbank.shape[1]//8+1
            fbank=pad_sequences(fbank,maxlen=self.audio_len,dtype='float',padding='post',truncating='post')
            fbank=fbank.reshape(fbank.shape[0],fbank.shape[1],1)
            input_lengths.append(input_length)
            wavs.append(fbank)
            label_length=len(label)
            label_lengths.append(label_length)
            label=np.array(label+[0]*(self.label_len-label_length))
            labels.append(label)
        if len(wavs)!=0:
            wavs=np.array(wavs)
            the_inputs=np.expand_dims(wavs,axis=4)
            the_inputs=wavs.transpose((0,2,1,3))
            the_labels=np.array(labels)
            input_lengths=np.array(input_lengths)
            label_lengths=np.array(label_lengths)
            inputs=[the_inputs,the_labels,input_lengths,label_lengths]
            outputs=np.zeros([len(wavs),1],dtype=np.float32)
            yield inputs,outputs


    def create_dict(self):
        self.py2id={}
        self.id2py={}
        with open(self.py2id_dict,'r',encoding='utf-8') as file:
            for line in file:
                py,idx=line.strip('\n').strip().split('\t')
                self.py2id[py.strip()]=int(idx.strip())
                self.id2py[int(idx.strip())]=py.strip()


def main():
    data=DataSpeech()
    data_iters=data.create_batch()
    for batch in data_iters:
        x,y=batch
        print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,y.shape)

if __name__=="__main__":
    main()
