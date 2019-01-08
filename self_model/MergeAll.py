import numpy as np
import tensorflow as tf
import os
from DataNon import DataSpeech
from DataNon import ConfigSpeech as sconfig
from SpeechModelForNon import ModelSpeech
from SpeechModelForNon import train as strain
from SpeechModelForNon import evaluate as sevaluate

from Language import ModelLanguage
from Language import train as ltrain
from Language import evaluate as sevaluate

from DataCBHG import ConfigLanguage as lconfig
from DataCBHG import DataLanguage

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
    lmodel=ModelLanguage()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        ckpt=tf.train.latest_checkpoint(lconfig.model_dir)
        if ckpt!=None:
            print('正在加载语言模型')
            saver.restore(sess,ckpt)
            while True:
                wav_path=input('请输入音频文件路径: ')
                if wav_path=='exit':break
                if not os.path.exists(wav_path):
                    print(wav_path+' 不存在')
                    continue
                pred,text=speech_online(wav_path)
                pred=pred.reshape(1,-1)
                feed={lmodel.x:pred}
                print(lmodel.is_training)
                predid=sess.run(lmodel.preds,feed_dict=feed)
                hzs=[lmodel.id2hz[idx] for idx in predid[0]]
                print(' '.join(text)+'\n')
                print(''.join(hzs)+'\n')





def main():
    merge_online()
    pass


if __name__=="__main__":
    main()
