#!/usr/bin/env python
# coding=utf-8

def scan_dict():
    data_path='/data/dataset/pinyin2hanzi/py2hz.tsv'
    pnys_list=[]
    hanzis_list=[]
    with open(data_path,'r',encoding='utf-8') as file:
        for line in file:
            idx,pnys,hanzis=line.strip('\n').strip().split('\t')
            pnys=[pny.strip() for pny in pnys.split(' ')]
            hanzis=[hanzi.strip() for hanzi in hanzis.split(' ')]
            pnys_list+=[pny for pny in pnys if pny not in pnys_list]
            hanzis_list+=[hanzi for hanzi in hanzis if hanzi not in hanzis_list]

    with open('/data/dataset/dict/raw_dict.txt','r',encoding='utf-8') as file:
        for line in file:
            py,hanzis=line.strip('\n').strip().split('\t')
            hanzis=[hanzi.strip() for hanzi in list(hanzis.strip())]
            if py not in pnys_list:pnys_list.append(py)
            hanzis_list+=[hanzi for hanzi in hanzis if hanzi not in hanzis_list]
    pnys_list=list(set(pnys_list))
    hanzis_list=list(set(hanzis_list))
    pnys_list=['<PAD>']+pnys_list+['-']
    hanzis_list=['<PAD>']+hanzis_list+['-']

    with open('/data/dataset/dict/py2id_dict.txt','w',encoding='utf-8') as file:
        for idx,pny in enumerate(pnys_list):
            file.write(pny+'\t'+str(idx)+'\n')
    with open('/data/dataset/dict/hz2id_dict.txt','w',encoding='utf-8') as file:
        for idx,hanzi in enumerate(hanzis_list):
            file.write(hanzi+'\t'+str(idx)+'\n')
def scan_py2hz():
    data_path='/data/dataset/pinyin2hanzi/py2hz.tsv'
    py2hz={}
    with open(data_path,'r',encoding='utf-8') as file:
        for line in file:
            idx,pnys,hanzis=line.strip('\n').strip().split('\t')
            pnys=[pny.strip() for pny in pnys.split(' ')]
            hanzis=[hanzi.strip() for hanzi in hanzis.split(' ')]
            assert len(pnys)==len(hanzis)
            for pny,hanzi in zip(pnys,hanzis):
                if pny not in py2hz.keys():
                    py2hz[pny]=[hanzi]
                elif hanzi not in py2hz[pny]:
                    py2hz[pny].append(hanzi)
    with open('/data/dataset/dict/raw_dict.txt','r',encoding='utf-8') as file:
        for line in file:
            py,hanzis=line.strip('\n').strip().split('\t')
            hanzis=[hanzi.strip() for hanzi in list(hanzis.strip())]
            if py in py2hz.keys():
                py2hz[py]+=hanzis
            else:
                py2hz[py]=hanzis
    py2hz['<PAD>']=['<PAD>']
    py2hz['_']=['_']
    with open('/data/dataset/dict/py2hz_dict.txt','w',encoding='utf-8') as file:
        for py,hanzis in py2hz.items():
            hanzis=' '.join(hanzis)
            file.write(py+'\t'+hanzis+'\n')
if __name__=='__main__':
    scan_dict()
    scan_py2hz()
