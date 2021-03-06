#!/usr/bin/env python
# coding=utf-8

import os
import random
import numpy as np
class DataConfig():
    base_dir='/data/dataset/'
    dict_dir=base_dir+'dict/'
    py2id_dict=dict_dir+'py2id_dict.txt'
    hz2id_dict=dict_dir+'hz2id_dict.txt'
    py2hz_dict=dict_dir+'py2hz_dict.txt'
    py2hz_dir=base_dir+'pinyin2hanzi/'
    types=['train','test','dev']
class ConfigLanguage(DataConfig):
    epochs=100
    model_dir='models/language_model/new/'
    model_name='model'
    model_path=model_dir+model_name
    embed_size=300
    num_hb=4
    num_eb=16
    norm_type='bn'
    lr=0.0001
    is_training=True
    batch_size=256
    py_size=1472
    hz_size=7459
    dropout_rate=0.5
class DataLanguage(ConfigLanguage):
    def __init__(self):
        super(DataLanguage,self).__init__()
        self.py2hz_paths={type:self.py2hz_dir+'py2hz_'+type+'.tsv' for type in self.types}
        self.create_dict()
        self.create_py2hz()
    def create_py2hz(self):
        self.py2hz={}
        self.batch_num={}
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
                batch_num=start_num//self.batch_size
                self.batch_num[_type]=batch_num
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
            assert len(py)==len(hz)
            if len(pys)==self.batch_size:
                inputs,outputs=self.seq_pad(pys,hzs)
                yield inputs,outputs
                pys,hzs=[],[]
            pys.append(py)
            hzs.append(hz)
    def create_online(self,text):
        pred=[self.py2id[py] for py in text]
        pred=np.array(pred)
        pred=pred.reshape((1,pred.shape[0]))
        return pred
    def seq_pad(self,pys,hzs):
        max_len=max([len(py) for py in pys])
        inputs=np.array([line+[0]*(max_len-len(line)) for line in pys])
        outputs=np.array([line+[0]*(max_len-len(line)) for line in hzs])
        return inputs,outputs

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
    data=DataLanguage()
    data_iters=data.create_batch()

    for batch in data_iters:
        x,y=batch
        print(x,'\n',y)


if __name__=="__main__":
    main()
