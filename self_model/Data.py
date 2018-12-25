#!/usr/bin/env python
# coding=utf-8

import os
class DataBase():
    def __init__(self):
        self.config()

    def config(self):
        self.base_dir='/data/dataset/'
        self.data_names=['aishell','st-cmds','primewords','thchs30']
        self.data_dirs={name:'/data/dataset/'+name+'/' for name in self.data_names}
        self.wav2py_paths={}
        self.types=['train','test','dev']
        self.batch_size=16
        for type in self.types:
            temp={}
            for name in ['syllabel','wav']:
                temp[name]=type+'.'+name+'.txt' if name!='wav' else type+'.'+name+'.lst'
            self.wav2py_paths[type]=temp
        self.dict_dir=self.base_dir+'dict/'
        self.py2id_dict=self.dict_dir+'py2id_dict.txt'
        self.hz2id_dict=self.dict_dir+'hz2id_dict.txt'
        self.py2hz_dict=self.dict_dir+'py2hz_dict.txt'
        
        self.py2hz_dir=self.base_dir+'pinyin2hanzi/'
        self.py2hz_paths={type:self.py2hz_dir+'py2hz_'+type+'.tsv' for type in self.types}
        assert os.path.exists(self.py2hz_paths['train'])
        print(self.py2hz_paths)
    def create_dict(self):
        self.py2id={}
        self.id2py={}
        self.hz2id={}
        self.id2hz={}
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


class DataSpeech(DataBase):
    def __init__(self):
        super(DataSpeech,self).__init__()
    def create_wav2py(self):
        self.wav2py={}
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
                    self.wav2py[_type][start_num+idx]={id2wav[key]:id2py[key]}
                start_num=len(self.wav2py[_type]) 
                print(_type+':'+str(start_num))                    
                
        print(self.wav2py['train'][1])
    def create_batch(selfi,flag='train',shuffle=True):
        data_num=len(self.wav2py[flag])
        if shuffle:
            idxs=random.shuffle(list(range(data_num)))
        else:
            idxs=list(range(data_num))
        wavs=[]
        labels=[]
        for i,idx in enumerte(idxs):
            wav_path,pys=self.wav2py[flag][idx].items()
            fbank=compute_fbank(wav_path)
            pad_fbank=np.zeros((fbank.shape[0]//8*8+8,fbank.shape[1]))
            pad_fbank[:fbank.shape[0],:]=fbank
            label=[self.py2id[py] for py in pys]
            assert len(wavs)==len(labels)
            if len(wavs)==self.batch_size:
                yield wavs,labels
                wavs,labels=[],[]
            wavs.append(pad_fbank)
            labels.append(label)
        if len(wavs)!=0:
            yield wavs,labels
            
class DataLanguage(DataBase):
    def __init__(self):
        super(DataLanguage,self).__init__()
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
        print(self.py2hz['train'][1])
    def create_batch(selfi,flag='train',shuffle=True):
        data_num=len(self.py2hz[flag])
        if shuffle:
            idxs=random.shuffle(list(range(data_num)))
        else:
            idxs=list(range(data_num))
        for i,idx in enumerte(idxs):
            fbank=compute_fbank(self.py2hz)


if __name__=="__main__":
    data=DataLanguage()
    data.create_py2hz()
