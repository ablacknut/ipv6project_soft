import os
import sys
# sys.path.append(r'D:\papers\恶意网站\项目\soft')
# os.chdir(r'D:\papers\恶意网站\项目\soft')
import keras.models
from keras.layers.core import *
from keras.models import *
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers import Input, concatenate, LSTM
from keras.layers import Conv1D, MaxPooling1D, TimeDistributed
from keras.layers import Dense, Flatten, Add
from keras.layers import Bidirectional, Embedding, BatchNormalization, AveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from algorithm.capsule_layer import CategoryCap, PrimaryCap
import jieba
import pickle
vali_rate = 0.2
EMBEDDING_DIM = 200
MAX_NB_WORDS = 0
MAX_SEQUENCE_LENGTH = 140
epoch = 20
T_fea = 'T7'
save_model_file = 'models/' + T_fea + 'best_MUlit140.h5'
Read_data = 'dataset_url/paper1_' + T_fea + '_data.csv'
Dictionary_Path2 = 'word2vec/paper1_' + T_fea + '_model_200.txt'
# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True  #allocate dynamically
# sess = tf.Session(config=config)


def MultiHeadsAttModel(l=8 * 8, d=512, dv=64, dout=512, nv=8):
    v1 = Input(shape=(l, d))
    q1 = Input(shape=(l, d))
    k1 = Input(shape=(l, d))

    v2 = Dense(dv * nv, activation="relu")(v1)
    q2 = Dense(dv * nv, activation="relu")(q1)
    k2 = Dense(dv * nv, activation="relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)

    att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[-1, -1]) / np.sqrt(dv), output_shape=(l, nv, nv))([q, k])  # l, nv, nv
    att = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(att)

    out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[4, 3]), output_shape=(l, nv, dv))([att, v])
    print('bbbbbbbbbbb', out)

    out = Reshape([l, d])(out)

    out = Add()([out, q1])

    out = Dense(dout, activation="relu")(out)

    return Model(inputs=[q1, k1, v1], outputs=out)


class NormL(Layer):
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', shape=(1, input_shape[-1]), initializer='ones', trainable=True)
        self.b = self.add_weight(name='kernel', shape=(1, input_shape[-1]), initializer='zeros', trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out * self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape


embedding_matrix_PATH = r'models/embedding_matrix_s.pkl'
tokenizer_PAYH = r'models/tokenizer_s.pkl'
with open(embedding_matrix_PATH, 'rb') as em, open(tokenizer_PAYH, 'rb') as t:
    embedding_matrix = pickle.load(em)
    tokenizer = pickle.load(t)

from algorithm.indrnn import IndRNN
sequence_input4 = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
wo_ind = Embedding(len(embedding_matrix),
                   EMBEDDING_DIM,
                   weights=[embedding_matrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(sequence_input4)
# wo_ind=Attention_layer()(wo_ind)
# print('Attention_layer:',wo_ind)
# wo_ind =Reshape((20,25,8))(wo_ind)
print('wo_ind:', wo_ind)

att = MultiHeadsAttModel(l=MAX_SEQUENCE_LENGTH, d=200, dv=25, dout=8, nv=8)
#l和d要和上一层维度相同， dv*nv=d
output = att([wo_ind, wo_ind, wo_ind])
print("x:", output)
# wo_ind = Reshape([83, 2, 32])(output)
# print("x:",wo_ind)

wo_ind = NormL()(output)

wo_ind = IndRNN(64, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                return_sequences=True)(wo_ind)
wo_ind = IndRNN(64, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                return_sequences=True)(wo_ind)
k = 12
# conv1= Conv1D(kernel_size=3, filters=k, padding='same',activation='tanh', strides=1)(wo_ind)
conv1 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1')(wo_ind)
num_capsule1 = 2
dim_capsule1 = 32
num_routing = 3
primary_caps = PrimaryCap(conv1,
                          dim_vector=dim_capsule1,
                          n_channels=32,
                          kernel_size=9,
                          strides=2,
                          padding='same',
                          name="primary_caps")
primary_caps = BatchNormalization()(primary_caps)
primary_caps = Dropout(0.3)(primary_caps)
# Layer 3: Capsule layer. Routing algorithm works here.
category_caps = CategoryCap(num_capsule=num_capsule1, dim_vector=dim_capsule1, num_routing=num_routing,
                            name='category_caps')(primary_caps)
output = BatchNormalization()(category_caps)
output = Flatten()(output)
# output = Dense(32, activation='relu')(output)
preds = Dense(1, activation='sigmoid')(output)
print("training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
model = Model(inputs=[sequence_input4], outputs=[preds])
model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])


def predict_url(urls):
    result = []
    _threshold = 0.5
    model.load_weights(save_model_file)
    texts = []
    for url in urls:
        words = jieba.cut(url)  ###
        tmp = [c for c in url]
        texts.append(list(words) + tmp)
    # return str(texts)
    se = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(se, maxlen=MAX_SEQUENCE_LENGTH)
    pred_class = model.predict([data]).reshape(len(urls))
    print(pred_class)
    for url, pred in zip(urls, pred_class):
        if pred > 0.5:
            result.append(url + '\t\t良性网站')
        else:
            result.append(url + '\t\t恶性网站')
    resultstr = '\n'.join(result)
    return resultstr


# train()

if __name__ == "__main__":
    urls = ['www.baidu.com', 'www.freebuf.com', 'ad-emea.doubleclick.net.1000.9007.302br.net']

    print(predict_url(urls))