import numpy as np
import tensorflow as tf
import os
from DataFix import DataSpeech
from DataFix import ConfigSpeech as sconfig
from SpeechModelForFix import ModelSpeech
from SpeechModelForFix import train as strain
from SpeechModelForFix import evaluate as sevaluate

from CBHG_self import read_dict,read_data,Graph
from CBHG_self import Config as lconfig

def read_wav(file_path):
    pass
def transform_wav(file_path):
    pass


def speech_online(wav_path):
    model=ModelSpeech()
    model.ctc_model.load_weights(sconfig.model_path)
    model.is_training=False
    batch=next(model.create_online(wav_path))
    result=model.predict_model.predict_on_batch(batch[0])
    pred,text=model.decode_ctc(result)
    return pred,text



def merge_online():
    lconfig.is_training=False
    _,_=read_data('train')
    py2id,id2py,hz2id,id2hz=read_dict()
    lmodel=Graph()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        ckpt=tf.train.latest_checkpoint(lconfig.model_dir)
        if ckpt!=None:
            print('正在加载语言模型')
            saver.restore(sess,ckpt)
            while True:
                wav_path=input("亲输入音频文件: ")
                if not os.path.exists(wav_path):
                    print('输入路径不存在')
                    continue
                pred,text=speech_online(wav_path)
                pred=pred.reshape(1,-1)
                feed={lmodel.x:pred}
                predid=sess.run(lmodel.preds,feed_dict=feed)
                hzs=[id2hz[idx] for idx in predid[0]]
                print(' '.join(text)+'\n')
                print(''.join(hzs)+'\n')


def main():
    merge_online()
if __name__=="__main__":
    main()
