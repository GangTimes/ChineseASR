import numpy as np
import tensorflow as tf
import os
from DataFix import DataSpeech
from DataFix import ConfigSpeech as sconfig
from SpeechModelForFix import ModelSpeech
from SpeechModelForFix import train as strain
from SpeechModelForFix import evaluate as sevaluate

from Language import ModelLanguage
from Language import train as ltrain
from Language import evaluate as sevaluate

from Data import ConfigLanguage as lconfig
from Data import DataLanguage

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

def language_online(pred):
    model=ModelLanguage()
    saver=tf.train.Saver()
    pred=np.array(pred)
    pred=pred.reshape((1,pred.shape[0]))
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt=tf.train.latest_checkpoint(lconfig.model_dir)
        if ckpt!=None:
            print("正在恢复模型")
            saver.restore(sess,ckpt)
            feed={model.x:pred}
            predid=sess.run(model.preds,feed_dict=feed)
            hzs=model.hz_decode(predid)
    text=''.join(hzs)
    return text

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
                pred,text=speech_online(wav_path)
                pys=lmodel.create_online(text)
                feed={lmodel.x:pys}
                predid=sess.run(lmodel.preds,feed_dict=feed)
                hzs=lmodel.hz_decode(predid)
                print(' '.join(text)+'\n')
                print(''.join(hzs)+'\n')





def main():
    merge_online()
    pass


if __name__=="__main__":
    main()
