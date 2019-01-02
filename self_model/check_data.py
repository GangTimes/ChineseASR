#!user/bin/env python3
# -*- coding:utf-8 -*-

from Utils import extract_feature
class config():
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
    audio_len=1600
    label_len=64
def create_wav2py():
    wav2py={}
    for _type,path in config.wav2py_paths.items():
        wav2py[_type]={}
        start_num=0
        for name,data_dir in config.data_dirs.items():
            id2wav={}
            id2py={}
            with open(data_dir+config.wav2py_paths[_type]['wav'],'r',encoding='utf-8') as file:
                for line in file:
                    idx,path=line.strip('\n').strip().split(' ')
                    id2wav[idx.strip()]=config.base_dir+path.strip()
            with open(data_dir+config.wav2py_paths[_type]['syllabel'],'r',encoding='utf-8') as file:
                for line in file:
                    ws=line.strip('\n').strip().split(' ')
                    idx,pys=ws[0],ws[1:]
                    id2py[idx.strip()]=pys
            assert len(id2py)==len(id2wav)
            for idx,key in enumerate(id2py.keys()):
                wav2py[_type][start_num+idx]=(id2wav[key],id2py[key])
            start_num=len(wav2py[_type])
    return wav2py
def check_audio(wav_path):
    fbank=extract_feature(wav_path)
    if fbank.shape[0]>config.audio_len:
        with open('log/check_audio.txt','+a') as file:
            file.write(wav_path+'\n')
def check_label(pys,wav_path):
    if len(pys)>config.label_len:
        with open('log/check_label.txt','+a') as file:
            file.write(wav_path+'\n')
def check_input(wav_path,label):
    fbank=extract_feature(wav_path)
    if (fbank.shape[0]//8)<len(label):
        with open('log/check_input.txt','+a') as file:
            file.write(wav_path+'\t'+'fbank_len:'+str(fbank.shape[0])+'\tinput_len'+str(fbank.shape[0]//8)+'\tlabel_len'+str(len(label))+'\n')
def report_data(wav_path,label):
    fbank=extract_feature(wav_path)
    with open('log/data_report.txt','+a') as file:
        file.write(wav_path+'\t'+'fbank_len:'+str(fbank.shape[0])+'\tinput_len:'+str(fbank.shape[0]//8)+'\tlabel_len:'+str(len(label))+'\n')
from tqdm import tqdm
def main():
    wav2py=create_wav2py()
    for _type in config.types:
        for idx in tqdm(range(len(wav2py[_type]))):
            wav_path,pys=wav2py[_type][idx]
            #report_data(wav_path,pys)
            #check_audio(wav_path)
            #check_label(pys,wav_path)
            check_input(wav_path,pys)
if __name__=="__main__":
    main()
