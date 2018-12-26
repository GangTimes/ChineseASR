#!/usr/bin/env python
# coding=utf-8

import os
import random
import numpy as np
from Utils import compute_fbank
class DataConfig():
    base_dir='/data/dataset/'
    data_names=['aishell','st-cmds','primewords','thchs30']
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
    label_max_len=64
    audio_len=1600
    audio_feature_len=200
    epochs=10
    save_step=1000
    batch_size=16
    model_dir='models/speech_model/'
    model_name='speechckpt'
    model_path=model_dir+model_name

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
        for i,idx in enumerate(idxs):
            wav_path,pys=self.wav2py[flag][idx]
            fbank=compute_fbank(wav_path)
            pad_fbank=fbank[:fbank.shape[0]//8*8,:]
            label=[self.py2id[py] for py in pys]
            assert len(wavs)==len(labels)
            if len(wavs)==self.batch_size:
                the_inputs,input_length=self.wav_padding(wavs)
                the_labels,label_length=self.label_padding(labels)
                inputs={'the_inputs':the_inputs,"the_labels":the_labels,"input_length":input_length,"label_length":label_length}
                outputs={'ctc':np.zeros(the_inputs.shape[0])}
                yield inputs,outputs
                wavs,labels=[],[]
            wavs.append(pad_fbank)
            labels.append(label)
        if len(wavs)!=0:
            the_inputs,input_length=self.wav_padding(wavs)
            the_labels,label_length=self.label_padding(labels)
            inputs={'the_inputs':the_inputs,"the_labels":the_labels,"input_length":input_length,"label_length":label_length}
            outputs={'ctc':np.zeros(the_inputs.shape[0])}
            yield inputs,outputs

    def wav_padding(self,wavs):
        wav_lens=[len(wav) for wav in wavs]
        max_len=max(wav_lens)
        wav_lens=np.array([leng//8 for leng in wav_lens])
        new_wavs=np.zeros((len(wavs),max_len,self.audio_feature_len,1))
        for i in range(len(wavs)):
            new_wavs[i,:wavs[i].shape[0],:,0]=wavs[i]
        return new_wavs,wav_lens

    def label_padding(self,labels):
        label_lens=np.array([len(label) for label in labels])
        max_len=max(label_lens)
        new_labels=np.zeros((len(labels),max_len))
        for i in range(len(labels)):
            new_labels[i,:len(labels[i])]=labels[i]
        return new_labels,label_lens
    def create_dict(self):
        self.py2id={}
        self.id2py={}
        with open(self.py2id_dict,'r',encoding='utf-8') as file:
            for line in file:
                py,idx=line.strip('\n').strip().split('\t')
                self.py2id[py.strip()]=int(idx.strip())
                self.id2py[int(idx.strip())]=py.strip()



class DataLanguage(DataConfig):
    def __init__(self):
        super(DataLanguage,self).__init__()
        self.py2hz_paths={type:self.py2hz_dir+'py2hz_'+type+'.tsv' for type in self.types}
        self.create_dict()
        self.create_py2hz()
    def create_py2hz(self):
        self.py2hz={}
        for _type,path in self.py2hz_paths.items():
            self.py2hz[_type]={}
            start_num=0
            with open(path,'r',encoding='utf-8') as file:
                for line in file:
                    idx,pys,hzs=line.strip('\n').strip().split('\t')
                    pys=pys.strip().split(' ')
                    hzs=hzs.strip().split(' ')
                    self.py2hz[_type][start_num]=(pys,hzs)
                    start_num+=1
                print(_type+':'+str(start_num))
    def create_batch(self,flag='train',shuffle=True):
        data_num=len(self.py2hz[flag])
        idxs=list(range(data_num))
        if shuffle:
            random.shuffle(idxs)
        pys=[]
        hzs=[]
        for i,idx in enumerate(idxs):
            py,hz=self.py2hz[flag][idx]
            py=[self.py2id[p] for p in py]
            hz=[self.hz2id[h] for h in hz]
            assert len(pys)==len(hzs)
            if len(pys)==self.batch_size:
                yield pys,hzs
                pys,hzs=[],[]
            pys.append(py)
            hzs.append(hz)
        if len(pys)!=0:
            yield pys,hzs
    def create_dict(self):
        self.py2id={}
        self.id2py={}
        self.id2hz={}
        self.hz2id={}
        with open(self.py2id_dict,'r',encoding='utf-8') as file:
            for line in file:
                py,idx=line.strip('\n').strip().split('\t')
                self.py2id[py.strip()]=int(idx.strip())
                self.id2py[int(idx.strip())]=py.strip()
        with open(self.hz2id_dict,'r',encoding='utf-8') as file:
            for line in file:
                hz,idx=line.strip('\n').strip().split('\t')
                self.hz2id[hz.strip()]=int(idx.strip())
                self.id2hz[int(idx.strip())]=hz.strip()

def main():
    data=DataSpeech()
    batch=data.create_batch()
    for b in batch:
        print(b)
        break
if __name__=="__main__":
    main()
