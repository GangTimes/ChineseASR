#!/usr/bin/env python
# coding=utf-8
import os
import tqdm
import numpy as np
import tensorflow as tf
gpu_id='2'
class Config:
    train_path='/data/dataset/pinyin2hanzi/py2hz_train.tsv'
    hz2id_dict='/data/dataset/dict/hz2id_dict.txt'
    test_path='/data/dataset/pinyin2hanzi/py2hz_test.tsv'
    dev_path='/data/dataset/pinyin2hanzi/py2hz_dev.tsv'
    py2id_dict='/data/dataset/dict/py2id_dict.txt'
    model_dir='log/CBHG_model/'
    model_name='model'
    model_path=model_dir+model_name
    board_path='tensorboard/CBHG'
    embed_size = 300
    num_highwaynet_blocks = 4
    encoder_num_banks = 8
    lr = 0.001
    is_training = True
    epochs = 25
    batch_size = 256

def read_dict():
    """
    根据路径dict_path读取文本和英文字典
    return: pny2idx idx2pny hanzi2idx idx2hanzi
    """
    pny2idx={}
    hanzi2idx={}
    idx2pny={}
    idx2hanzi={}
    with open(Config.hz2id_dict,'r',encoding='utf-8') as file:
        for line in file:
            hanzi,idx=line.strip('\n').split('\t')
            hanzi2idx[hanzi]=int(idx.strip())
            idx2hanzi[int(idx.strip())]=hanzi.strip()
    with open(Config.py2id_dict,'r',encoding='utf-8') as file:
        for line in file:
            pny,idx=line.strip('\n').split('\t')
            pny2idx[pny]=int(idx.strip())
            idx2pny[int(idx.strip())]=pny.strip()
        
    Config.pny_size=len(pny2idx)
    Config.hanzi_size=len(hanzi2idx)
    return pny2idx,idx2pny,hanzi2idx,idx2hanzi


def read_data(type):
    """
    根据路径data_path读取中文文本到英文文本的对应关系  
    return: inputs->拼音->[[一句话的拼音列表],[]]  lables->汉字->[[一句话的汉字列表],[]]
    """
    inputs=[]
    labels=[]
    if type=='train':
        data_path=Config.train_path
    elif type=='test':
        data_path=Config.test_path
    elif type=='dev':
        data_path=Config.dev_path
    else:
        raise Exception("Invalid type!", type)

    with open(data_path,'r',encoding='utf-8') as file:
        for line in file:
            key,pny,hanzi=line.strip('\n').strip().split('\t')
            pnys=pny.strip().split(' ')
            hanzis=hanzi.strip().split(' ')
            
            assert len(pnys)==len(hanzis)
            inputs.append(pnys)
            labels.append(hanzis)
        
    pny2idx,idx2pny,hanzi2idx,idx2hanzi=read_dict()
    input_num = [[pny2idx[pny] for pny in line ] for line in inputs]
    label_num = [[hanzi2idx[han] for han in line] for line in labels]

    return input_num,label_num

def get_batch(inputs,labels):
    batch_size=Config.batch_size
    batch_num = len(inputs) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        input_batch = inputs[begin:end]
        label_batch = labels[begin:end]
        max_len = max([len(line) for line in input_batch])
        input_batch = np.array([line + [0] * (max_len - len(line)) for line in input_batch])
        label_batch = np.array([line + [0] * (max_len - len(line)) for line in label_batch])
        yield input_batch, label_batch


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)

def prenet(inputs, num_units=None, is_training=True, scope="prenet", reuse=None, dropout_rate=0.2):
    '''
    inputs: batch_size*length*embed_size
    return:batch_size*length*num_units/2
    '''

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training, name="dropout2")
    return outputs  # (N, ..., num_units[1])



def conv1d(inputs,filters=None, size=1,rate=1, padding="SAME",use_bias=False,activation_fn=None, scope="conv1d", reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''    
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "activation": activation_fn,
                  "use_bias": use_bias, "reuse": reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs



def conv1d_banks(inputs, num_units=None, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    '''Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.

    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, num_units // 2, 1)  # k=1
        for k in range(2, K + 1):  # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, num_units, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = normalize(outputs, is_training=is_training,
                            activation_fn=tf.nn.relu)
    return outputs  # (N, T, Hp.embed_size//2*K)


def conv1d_banks(inputs, num_units=None, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    '''Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.

    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, num_units // 2, 1)  # k=1
        for k in range(2, K + 1):  # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, num_units, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = normalize(outputs, is_training=is_training,
                            activation_fn=tf.nn.relu)
    return outputs  # (N, T, Hp.embed_size//2*K)



def gru(inputs, num_units=None, bidirection=False, seqlen=None, scope="gru", reuse=None):
    '''Applies a GRU.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results
        are concatenated.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        cell = tf.contrib.rnn.GRUCell(num_units)
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs,
                                                         sequence_length=seqlen,
                                                         dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
                                           sequence_length=seqlen,
                                           dtype=tf.float32)

    return outputs

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387
    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        C = 1. - T
        outputs = H * T + inputs * C


    return outputs


def normalize(inputs,
              decay=.99,
              epsilon=1e-8,
              is_training=True,
              activation_fn=None,
              reuse=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension. Note that this is different from the native
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py`
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
    # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    inputs = tf.expand_dims(inputs, axis=1)
    outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                            decay=decay,
                                            center=True,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=is_training,
                                            scope=scope,
                                            zero_debias_moving_mean=True,
                                            fused=True,
                                            reuse=reuse)
    outputs = tf.squeeze(outputs, axis=1)

    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs


class Graph():
    '''Builds a model graph'''

    def __init__(self):
        tf.reset_default_graph()
        self.pny_size = Config.pny_size
        self.han_size = Config.hanzi_size
        self.embed_size = Config.embed_size
        self.is_training = Config.is_training
        self.num_highwaynet_blocks = Config.num_highwaynet_blocks
        self.encoder_num_banks = Config.encoder_num_banks
        self.lr = Config.lr
        
        self.x = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))
        
        # Character Embedding for x

        enc = embed(self.x, self.pny_size, self.embed_size, scope="emb_x")
        # Encoder pre-net
        prenet_out = prenet(enc,
                            num_units=[self.embed_size, self.embed_size // 2],
                            is_training=self.is_training)  # (N, T, E/2)

        # Encoder CBHG
        ## Conv1D bank
        enc = conv1d_banks(prenet_out,
                            K=self.encoder_num_banks,
                            num_units=self.embed_size // 2,
                            is_training=self.is_training)  # (N, T, K * E / 2)

        ## Max pooling
        enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

        ## Conv1D projections
        enc = conv1d(enc, self.embed_size // 2, 5, scope="conv1d_1")  # (N, T, E/2)
        enc = normalize(enc, is_training=self.is_training,
                            activation_fn=tf.nn.relu, scope="norm1")
        enc = conv1d(enc, self.embed_size // 2, 5, scope="conv1d_2")  # (N, T, E/2)
        enc = normalize(enc, is_training=self.is_training,
                            activation_fn=None, scope="norm2")
        enc += prenet_out  # (N, T, E/2) # residual connections

        ## Highway Nets
        for i in range(self.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=self.embed_size // 2,
                                scope='highwaynet_{}'.format(i))  # (N, T, E/2)

        ## Bidirectional GRU
        enc = gru(enc, self.embed_size // 2, True, scope="gru1")  # (N, T, E)

        ## Readout
        self.outputs = tf.layers.dense(enc, self.han_size, use_bias=False)
        self.preds = tf.to_int32(tf.argmax(self.outputs, axis=-1))

        if self.is_training:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
            self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y)))  # masking
            self.hits = tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
            self.acc = tf.reduce_sum(self.hits) / tf.reduce_sum(self.istarget)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / tf.reduce_sum(self.istarget)

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar('mean_loss', self.mean_loss)
            tf.summary.scalar('acc', self.acc)
            self.merged = tf.summary.merge_all()


def train():
    inputs,labels=read_data('train')
    dev_inputs,dev_labels=read_data('dev')
    g = Graph()
    config=tf.ConfigProto(log_device_placement=True)
    saver =tf.train.Saver()
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        ckpt=tf.train.latest_checkpoint(Config.model_dir)
        if ckpt!=None:
            print("正在恢复模型")
            saver.restore(sess, ckpt)
        writer = tf.summary.FileWriter(Config.board_path, tf.get_default_graph())
        
        batch_num = len(inputs) // Config.batch_size
        dev_num=len(dev_inputs)//Config.batch_size
        for k in range(Config.epochs):
            total_loss = 0
            batch = get_batch(inputs, labels)
            for i in range(batch_num):
                input_batch, label_batch = next(batch)
                feed = {g.x: input_batch, g.y: label_batch}
                cost,_ = sess.run([g.mean_loss,g.train_op], feed_dict=feed)
                total_loss += cost
                if (k * batch_num + i) % 10 == 0:
                    rs=sess.run(merged, feed_dict=feed)
                    writer.add_summary(rs, k * batch_num + i)
            dev_batch=get_batch(dev_inputs,labels)
            for i in range(dev_num):
                dev_inputs_batch,dev_labels_batch=next(dev_batch)
                preds=sess.run(g.preds,{g.x:dev_inputs_batch})
                

            
            print('epochs', k+1, ': average loss = ', total_loss/batch_num)
            saver.save(sess,Config.model_path)
        writer.close()

def test():
    Config.is_training=False
    _,_=read_data('train')
    pny2id,id2pny,han2id,id2han=read_dict()
    g=Graph()
    config=tf.ConfigProto(log_device_placement=True)
    saver =tf.train.Saver()
    with tf.Session(config=config) as sess:
        ckpt=tf.train.latest_checkpoint(Config.model_dir)
        if ckpt!=None:
            print("正在恢复模型")
            saver.restore(sess, ckpt)
        while True:
            line = input('输入测试拼音: ')
            if line == 'exit': break
            line = line.strip('\n').split(' ')
            x = np.array([pny2id[pny] for pny in line])
            x = x.reshape(1, -1)
            preds = sess.run(g.preds, {g.x: x})
            got = ''.join(id2han[idx] for idx in preds[0])
            print(got)


if __name__=="__main__":
    #train()
    test()
