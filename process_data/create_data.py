#!/usr/bin/env python
# coding=utf-8
'''
提取aishell_data中的tar文件
'''
import os
def tar_aishell():
    path="/data/dataset/data_aishell/wav"
    files=[path+'/'+name for name in os.listdir(path)]
    for file in files:
        os.system("tar -xzvf "+file+" /data/dataset/data_aishell/")

def scan_aishell():
    """
    return pnys,hanzis,phdict
    """
    print('ai-shell')
    train_label='/data/dataset/aishell/train.syllabel.txt'
    dev_label='/data/dataset/aishell/dev.syllabel.txt'
    test_label='/data/dataset/aishell/test.syllabel.txt'
    labels=[train_label,dev_label,test_label]
    txt_path='/data/dataset/data_aishell/transcript/aishell_transcript_v0.8.txt'

    labels=[train_label,dev_label,test_label]
    id2text={}
    with open(txt_path,'r',encoding='utf-8') as text:
        for line in text:
            line_text=line.strip('\n').strip().split(' ')
            idx,hanzis=line_text[0],''.join(line_text[1:])
            
            idx=idx.strip()
            hanzis=[h.strip() for h in hanzis]
            id2text[idx]=list(hanzis)
    id2pny={}
    for label in labels:
        with open(label,'r',encoding='utf-8') as file:
            for line in file:
                line_text=line.strip('\n').strip().split(' ')
                idx,pny=line_text[0],line_text[1:]
                idx=idx.strip()
                pny=[p.strip() for p in pny]
                id2pny[idx]=pny
    
    assert len(id2pny)==len(id2text)
    with open('/data/dataset/pinyin2hanzi/aishell.tsv','w+',encoding='utf-8') as file:
        for idx in id2text.keys():
            file.write(idx+'\t'+' '.join(id2pny[idx])+'\t'+' '.join(id2text[idx])+'\n')


def scan_dict():
    paths=['aishell','thchs30','primewords','st-cmds']
    types=['train','test','dev']
    pnys=[]
    for path in paths:
        for type in types:
            file_path=path+'/'+type+'.syllabel.txt'
            assert os.path.exists(file_path)
            with open(file_path,'r',encoding='utf-8') as file:
                for line in file:
                    line_text=line.strip('\n').strip().split(' ')
                    idx,pny=line_text[0],line_text[1:]
                    idx=idx.strip()
                    pny=[p.strip() for p in pny]
                    pnys+=pny
    with open('dict.txt','r',encoding='utf-8') as file:
        for line in file:
            pny=line.strip('\n').strip().split('\t')[0].strip()
            pnys.append(pny)
    pnys=['<PAD>']+['_']+list(set(pnys))
    print("拼音字典长度:",len(pnys))
    with open('pndict.txt','w',encoding='utf-8') as file:
        for idx,pny in enumerate(pnys):
            file.write(str(idx)+'\t'+pny+'\n')


def scan_thchs30():
    """
    return pnys,hanzis,phdict
    """
    data_dir='/data/dataset/thchs30'
    types=['train','test','dev']

    id2text={}
    id2pny={}
    for type in types:
        label_path=data_dir+'/'+type+'.syllabel.txt'
        with open(label_path,'r',encoding='utf-8') as text:
            for line in text:
                line_pny=line.strip('\n').strip().split(' ')
                idx,pny=line_pny[0],line_pny[1:]
                
                idx=idx.strip()
                pny=[p.strip() for p in pny]
                id2pny[idx]=list(pny)
    tr_path=os.listdir('/data/dataset/data_thchs30/data/')
    tr_path=[path for path in tr_path if 'trn' in path]
    tr_path={path.split('.')[0]:path for path in tr_path}
    for idx in id2pny.keys():
        with open('/data/dataset/data_thchs30/data/'+tr_path[idx],'r',encoding='utf-8') as file:
            line=file.readlines()[0].strip('\n').strip()
            line=[l.strip() for l in line]
            line=list(''.join(line))
            assert len(id2pny[idx])==len(line),print(idx,id2pny[idx],line)
            id2text[idx]=line
    
    assert len(id2pny)==len(id2text)
    with open('/data/dataset/pinyin2hanzi/thchs30.tsv','w+',encoding='utf-8') as file:
        for idx in id2text.keys():
            file.write(idx+'\t'+' '.join(id2pny[idx])+'\t'+' '.join(id2text[idx])+'\n')

def scan_primewords():
    data_dir='/data/dataset/primewords/'
    types=['train','test','dev']
    id2pny={}
    for type in types:
        label_path=data_dir+type+'.syllabel.txt'
        with open(label_path,'r',encoding='utf-8') as text:
            for line in text:
                line_pny=line.strip('\n').strip().split(' ')
                idx,pny=line_pny[0],line_pny[1:]
                
                idx=idx.strip()
                pny=[p.strip() for p in pny]
                id2pny[idx]=list(pny)
    import json
    with open('/dataset/dataset/primewords_md_2018_set1/set1_transcript.json','r') as file:
        data=json.load(file)
    id2text={}
    count=0
    for d in data:
        line=d['text'].strip()
        idx=d['id'].strip()
        line=[l.strip() for l in line if l not in [',','，',':','：',' ?']]
        line=list(''.join(line))
        if len(id2pny[idx])!=len(line):
            count+=1
            continue
        id2text[idx]=line
    print("放弃了"+str(count)+'个样本')
    with open('/data/dataset/pinyin2hanzi/primwords.tsv','w+',encoding='utf-8') as file:
        for idx in id2text.keys():
            file.write(idx+'\t'+' '.join(id2pny[idx])+'\t'+' '.join(id2text[idx])+'\n')

        
def scan_st_cmds():
    """
    return pnys,hanzis,phdict
    """
    data_dir='/data/dataset/ST-CMDS-20170001_1-OS/'
    types=['train','test','dev']

    id2text={}
    id2pny={}
    for type in types:
        label_path='st-cmds/'+type+'.syllabel.txt'
        with open(label_path,'r',encoding='utf-8') as text:
            for line in text:
                line_pny=line.strip('\n').strip().split(' ')
                idx,pny=line_pny[0],line_pny[1:]
                
                idx=idx.strip()
                pny=[p.strip() for p in pny]
                id2pny[idx]=list(pny)
    tr_path=os.listdir(data_dir)
    tr_path=[path for path in tr_path if '.txt' in path]
    tr_path={path.split('.')[0]:path for path in tr_path}
    count=0
    for idx in id2pny.keys():
        with open(data_dir+tr_path[idx],'r',encoding='utf-8') as file:
            line=file.readlines()[0].strip('\n').strip()
            line=[l.strip() for l in line if l not in [',','，',':','：',' ?']]
            line=list(''.join(line))
            if len(id2pny[idx])!=len(line):
                count+=1
                continue
            id2text[idx]=line
    
    print("放弃了"+str(count)+'个样本')
    with open('/data/dataset/pinyin2hanzi/st_cmds.tsv','w+',encoding='utf-8') as file:
        for idx in id2text.keys():
            file.write(idx+'\t'+' '.join(id2pny[idx])+'\t'+' '.join(id2text[idx])+'\n')


def merge_all():
    pys=['a','b']
    import random
    paths=['st_cmds','thchs30','primewords','aishell','lcqmc']
    paths=['/data/dataset/pinyin2hanzi/'+path+'.tsv' for path in paths]
    with open('/data/dataset/pinyin2hanzi/py2hz.tsv','w',encoding='utf-8') as writer:
        for path in paths:
            with open(path,'r' , encoding='utf-8') as file:
                for line in file:
                    idx,pnys,hanzis=line.strip('\n').strip().split('\t')
                    assert len(pnys.strip().split(' '))==len(hanzis.strip().split(' '))
                    temp=random.random()
                    if temp>0.1:
                        with open('/data/dataset/pinyin2hanzi/py2hz_train.tsv','a+',encoding='utf-8') as train:
                            train.write(idx+'\t'+pnys+'\t'+hanzis+'\n')
                    elif temp<0.05:
                        with open('/data/dataset/pinyin2hanzi/py2hz_test.tsv','a+',encoding='utf-8') as test:
                            test.write(idx+'\t'+pnys+'\t'+hanzis+'\n')
                    else:
                         with open('/data/dataset/pinyin2hanzi/py2hz_dev.tsv','a+',encoding='utf-8') as dev:
                            dev.write(idx+'\t'+pnys+'\t'+hanzis+'\n')
                    writer.write(idx+'\t'+pnys+'\t'+hanzis+'\n')

def scan_LCQMC():
    data_dir='/data/dataset/LCQMC/'
    paths=['train','dev','test']
    paths=[data_dir+'LCQMC_'+path+'.json' for path in paths]
    import json
    import re
    from pypinyin import pinyin,Style
    datas=[]
    with open('/data/dataset/pinyin2hanzi/lcqmc.tsv','w',encoding='utf-8') as writer:
        for path in paths:
            with open(path,'r',encoding='utf-8') as file:
                for line in file:
                    line=json.loads(line)
                    idx=line['ID'].strip()
                    text1=line['sentence1']
                    text2=line['sentence2']
                    pattern =re.compile(u"[\u4e00-\u9fa5]+")
                    result1=re.findall(pattern,text1)
                    result2=re.findall(pattern,text2)
                    # print result.group()
                    result=result1+result2
                    count=0
                    for w in result:
                        w=w.strip()
                        if w=='':
                            continue
                        sent=pinyin(w,style=Style.TONE3)
                        sent=[s[0].strip() if s[0][-1].isdigit() else s[0].strip()+'5'  for s in sent]
                        assert len(sent)==len(w)
                        sent=' '.join(sent)
                        w=' '.join(list(w))
                        
                        count+=1
                        writer.write(idx+str(count)+'\t'+sent+'\t'+w+'\n')

                
def scan_all():
    #scan_aishell()
    #scan_thchs30()
    #scan_st_cmds()
    #scan_primewords()
    #scan_LCQMC()
    merge_all()
if __name__=="__main__":
    scan_all()
