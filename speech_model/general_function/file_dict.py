#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
获取符号字典列表的程序
'''

def GetSymbolList(self):
    '''
    加载拼音符号列表，用于标记符号
    返回一个列表list类型变量
    '''
    id2py_dict={}
    with open('/data/dataset/dict/py2id_dict.txt','r',encoding='UTF-8') as file:
        for line in file:
            py,idx=line.strip('\n').strip().split('\t')
            id2py_dict[int(idx)]=py
    return id2py_dict
