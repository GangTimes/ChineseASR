import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc
from keras.preprocessing.sequence import pad_sequences



# 对音频文件提取mfcc特征
def compute_mfcc(file):
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)
    mfcc_feat = mfcc_feat[::3]
    mfcc_feat = np.transpose(mfcc_feat)
    mfcc_feat = pad_sequences(mfcc_feat, maxlen=500, dtype='float', padding='post', truncating='post').T
    return mfcc_feat


# 获取信号的时频图
def compute_fbank(file):
	x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
	fs, wavsignal = wav.read(file)
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
	wav_arr = np.array(wavsignal)
	wav_length = len(wavsignal)
	#print(wav_arr.shape)
	#wav_length = wav_arr.shape[1]
	range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
	data_line = np.zeros((1, 400), dtype = np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = wav_arr[p_start:p_end]
		data_line = data_line * w # 加窗
		data_line = np.abs(fft(data_line)) / wav_length
		data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	#print(data_input.shape)
	data_input = np.log(data_input + 1)
	data_input = data_input[::]
	data_input = np.transpose(data_input)
	#data_input = pad_sequences(data_input, maxlen=800, dtype='float', padding='post', truncating='post').T
	return data_input
import difflib
def get_edit_distance(label,pred):
    leven_cost=0
    s=difflib.SequenceMatcher(None,label,pred)
    for tag,i1,i2,j1,j2 in s.get_opcodes():
        if tag=='replace':
            leven_cost+=max(i2-i1,j2-j1)
        elif tag=='insert':
            leven_cost+=(j2-j1)
        elif tag=='delete':
            leven_cost+=(i2-i1)
    return leven_cost

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from scipy.fftpack import fft
import wave
def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    #wave_data = wave_data
    return wave_data, framerate

def extract_mfccfeature(wavsignal, fs):
    # 获取输入特征
    feat_mfcc=mfcc(wavsignal[0],fs)
    feat_mfcc_d=delta(feat_mfcc,2)
    feat_mfcc_dd=delta(feat_mfcc_d,2)
    # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature

def extract_freqfeature(wavsignal, fs):
    # wav波形 加时间窗以及时移10ms
    x=np.linspace(0,400-1,400,dtype=np.int64)
    w=0.54-0.46*np.cos(2*np.pi*(x)/(400-1))
    time_window = 25 # 单位ms
    window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值

    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]

    range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype = np.float)

    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[0, p_start:p_end]
        data_line = data_line * w # 加窗
        data_line = np.abs(fft(data_line)) / wav_length
        data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    return data_input
def extract_feature(file_path):
    wav_data,fs=read_wav_data(file_path)
    wav_feature=extract_freqfeature(wav_data,fs)
    wav_feature=wav_feature.reshape(wav_feature.shape[0],wav_feature.shape[1],1)
    return wav_feature
