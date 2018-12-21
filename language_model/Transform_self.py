#!/usr/bin/env python
# coding=utf-8
import os
import tqdm
import numpy as np
import tensorflow as tf

class Config:
    data_path='/data/dataset/pinyin2hanzi/py2hz_train.tsv'
    hz2id_dict='/data/dataset/dict/hz2id_dict.txt'
    py2id_dict='/data/dataset/dict/py2id_dict.txt'
    model_dir='log/transform_model/'                                           
    model_name='model'
    model_path=model_dir+model_name

    board_path='tensorboard/transform/'
    embed_size = 300
    lr = 0.0003
    is_training = True
    epochs = 125
    batch_size = 256
    num_heads = 8
    num_blocks = 6
    input_vocab_size = 50
    label_vocab_size = 50
    # embedding size
    max_length = 100
    hidden_units = 512
    dropout_rate = 0.2

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
        
    Config.input_vocab_size=len(pny2idx)
    Config.label_vocab_size=len(hanzi2idx)
    return pny2idx,idx2pny,hanzi2idx,idx2hanzi
def read_data():
    """
    根据路径data_path读取中文文本到英文文本的对应关系  
    return: inputs->拼音->[[一句话的拼音列表],[]]  lables->汉字->[[一句话的汉字列表],[]]
    """
    inputs=[]
    labels=[]
    hanzid=[]
    pnysd=[]
    phdict={}
    with open(Config.data_path,'r',encoding='utf-8') as file:
        for line in file:
            key,pny,hanzi=line.strip('\n').strip().split('\t')
            pnys=pny.strip().split(' ')
            hanzis=hanzi.strip().split(' ')
            
            inputs.append(pnys)
            labels.append(hanzis)
            assert len(pnys)==len(hanzis)
        
    pny2idx,idx2pny,hanzi2idx,idx2hanzi=read_dict()
    input_num = [[pny2idx[pny] for pny in line ] for line in inputs]
    label_num = [[hanzi2idx[han] for han in line] for line in labels]

    return input_num,label_num


def get_batch(input_data, label_data ):
    batch_size=Config.batch_size
    batch_num = len(input_data) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        input_batch = input_data[begin:end]
        label_batch = label_data[begin:end]
        max_len = max([len(line) for line in input_batch])
        input_batch = np.array([line + [0] * (max_len - len(line)) for line in input_batch])
        label_batch = np.array([line + [0] * (max_len - len(line)) for line in label_batch])
        yield input_batch, label_batch


def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5) 

    return outputs


def multihead_attention(emb,
                        queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs


def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


class Graph():
    def __init__(self, is_training=True):
        tf.reset_default_graph()
        self.is_training = Config.is_training
        self.hidden_units = Config.hidden_units
        self.input_vocab_size = Config.input_vocab_size
        self.label_vocab_size = Config.label_vocab_size
        self.num_heads = Config.num_heads
        self.num_blocks = Config.num_blocks
        self.max_length = Config.max_length
        self.lr = Config.lr
        self.dropout_rate = Config.dropout_rate
        
        # input
        self.x = tf.placeholder(tf.int32, shape=(None, None))
        self.y = tf.placeholder(tf.int32, shape=(None, None))
        # embedding
        self.emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True, scope="enc_embed")
        self.enc = self.emb + embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=self.max_length,num_units=self.hidden_units, zero_pad=False, scale=False,scope="enc_pe")
        ## Dropout
        self.enc = tf.layers.dropout(self.enc, 
                                    rate=self.dropout_rate, 
                                    training=self.is_training)
    
                
        ## Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                self.enc = multihead_attention(emb = self.emb,
                                               queries=self.enc, 
                                                keys=self.enc, 
                                                num_units=self.hidden_units, 
                                                num_heads=self.num_heads, 
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False)
                        
        ### Feed Forward
        self.outputs = feedforward(self.enc, num_units=[4*self.hidden_units, self.hidden_units])
            
                
        # Final linear projection
        self.logits = tf.layers.dense(self.outputs, self.label_vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)
                
        if is_training:  
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
            # Summary 
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()


def train():
    input_num,label_num=read_data()
    batch_size=Config.batch_size
    g = Graph()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2' #use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamicall
    saver =tf.train.Saver()
    with tf.Session(config=config) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        ckpt=tf.train.latest_checkpoint(Config.model_dir)
        if ckpt!=None:
            print("正在恢复模型")
            saver.restore(sess, ckpt)
        writer = tf.summary.FileWriter(Config.board_path, tf.get_default_graph())
        for k in range(Config.epochs):
            total_loss = 0
            batch_num = len(input_num) // batch_size
            batch = get_batch(input_num, label_num)
            for i in range(batch_num):
                input_batch, label_batch = next(batch)
                feed = {g.x: input_batch, g.y: label_batch}
                cost,_ = sess.run([g.mean_loss,g.train_op], feed_dict=feed)
                total_loss += cost
                if (k * batch_num + i) % 10 == 0:
                    rs=sess.run(merged, feed_dict=feed)
                    writer.add_summary(rs, k * batch_num + i)
            print('epochs', k+1, ': average loss = ', total_loss/batch_num)
            saver.save(sess, Config.model_path)
        writer.close()

def test():
    Config.is_training = False
    os.environ["CUDA_VISIBLE_DEVICES"] = '2' #use GPU with ID=0
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamicall
    pny2id,id2pny,han2id,id2han=read_dict()

    g = Graph()

    saver =tf.train.Saver()

    with tf.Session() as sess:
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
    train()
    #test()
