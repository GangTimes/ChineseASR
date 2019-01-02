#!/usr/bin/env python
# coding=utf-8
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from Data import DataLanguage
from Data import ConfigLanguage as config
import time
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


class ModelLanguage(DataLanguage):
    '''Builds a model graph'''

    def __init__(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))

        # Character Embedding for x

        enc = embed(self.x, self.py_size, self.embed_size, scope="emb_x")
        # Encoder pre-net
        prenet_out = prenet(enc,
                            num_units=[self.embed_size, self.embed_size // 2],
                            is_training=self.is_training)  # (N, T, E/2)

        # Encoder CBHG
        ## Conv1D bank
        enc = conv1d_banks(prenet_out,
                            K=self.num_eb,
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
        for i in range(self.num_hb):
            enc = highwaynet(enc, num_units=self.embed_size // 2,
                                scope='highwaynet_{}'.format(i))  # (N, T, E/2)

        ## Bidirectional GRU
        enc = gru(enc, self.embed_size // 2, True, scope="gru1")  # (N, T, E)

        ## Readout
        self.outputs = tf.layers.dense(enc, self.hz_size, use_bias=False)
        self.preds = tf.to_int32(tf.argmax(self.outputs, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y)))  # masking
        self.hits = tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
        self.acc = tf.reduce_sum(self.hits) / tf.reduce_sum(self.istarget)

        if self.is_training:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / tf.reduce_sum(self.istarget)

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)



def train(model=None,data=None):
    if model==None:
        model=ModelLanguage()
    if data==None:
        data=DataLanguage()
    saver =tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt=tf.train.latest_checkpoint(config.model_dir)
        if ckpt!=None:
            print("正在恢复模型")
            saver.restore(sess, ckpt)

        for epoch in range(1,config.epochs+1):
            start=time.time()
            total_loss = 0
            total_acc=0
            data_iters = data.create_batch('train',True)
            for i in tqdm(range(data.batch_num['train'])):
                input_batch, label_batch = next(data_iters)
                feed = {model.x: input_batch, model.y: label_batch}
                cost,acc,_ = sess.run([model.mean_loss,model.acc,model.train_op], feed_dict=feed)
                total_loss += cost
                total_acc+=acc
            print('epochs',epoch, ': average loss = ', total_loss/data.batch_num['train'])
            print('epochs',epoch, ': train accuracy = ', total_acc/data.batch_num['train'])
            evaluate(sess,model,data)
            saver.save(sess,config.model_path)
            end=time.time()
            print('Epoch:{:d}/{:d}===总共耗时{:.2f}秒'.format(epoch,config.epochs,(end-start)))
def evaluate(sess,model,data=None):
    if data==None:
        data=DataLanguage()
    model.is_training=False
    data_iters=data.create_batch('dev',False)
    total_acc=0
    for i in tqdm(range(data.batch_num['dev'])):
        input_batch,label_batch=next(data_iters)
        feed={model.x:input_batch,model.y:label_batch}
        acc= sess.run(model.acc, feed_dict=feed)
        total_acc+=acc
    print("验证accuracy:{.2f}".format(total_acc/data.batch_num['dev']))

def test():
    pass
def main():
    model=ModelLanguage()
    data=DataLanguage()
    train()
if __name__=="__main__":
    main()
