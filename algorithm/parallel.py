import os
import sys
import keras.models
from keras.layers.core import *
from keras.models import *
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers import Input, concatenate, LSTM
from keras.layers import Conv1D, MaxPooling1D, TimeDistributed
from keras.layers import Dense, Flatten
from keras.layers import Bidirectional, Embedding, BatchNormalization, AveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

#词向量化
MAX_NB_WORDS = 0
MAX_SEQUENCE_LENGTH2 = 115
MAX_SEQUENCE_LENGTH4 = 70
vali_rate = 0.2
EMBEDDING_DIM = 200
epoch = 5
T_fea1 = 'T6'
T_fea2 = 'T4'
T_model = 'our'
save_model_file = 'models/' + T_fea1 + 'best_115' + T_model + '.h5'
Read_data = 'dataset_url/paper2_' + T_fea1 + '_data.csv'
Dictionary_Path2 = 'word2vec/paper2_' + T_fea1 + '_model_200.txt'
Read_data4 = 'dataset_url/paper2_' + T_fea2 + '_data.csv'
Dictionary_Path4 = 'word2vec/paper2_' + T_fea2 + '_model_200.txt'

embedding_matrix_PATH = r'models/embedding_matrix_p.pkl'
# word_index_PATH = r'models/word_index_p.pkl'
tokenizer_PAYH = r'models/tokenizer_p.pkl'
embedding_matrix4_PATH = r'models/embedding_matrix4_p.pkl'
# word_index4_PATH = r'models/word_index4_p.pkl'
tokenizer4_PAYH = r'models/tokenizer4_p.pkl'
with open(embedding_matrix_PATH, 'rb') as em, open(embedding_matrix4_PATH,
                                                   'rb') as em4, open(tokenizer_PAYH, 'rb') as t, open(tokenizer4_PAYH,
                                                                                                       'rb') as t4:
    embedding_matrix = pickle.load(em)
    # word_index = pickle.load(wi)
    embedding_matrix4 = pickle.load(em4)
    # word_index4 = pickle.load(wi4)
    tokenizer = pickle.load(t)
    tokenizer4 = pickle.load(t4)

################################################################################模型部分
from algorithm.indrnn import IndRNN
sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH2, ), dtype='int32')
wo_ind = Embedding(len(embedding_matrix),
                   EMBEDDING_DIM,
                   weights=[embedding_matrix],
                   input_length=MAX_SEQUENCE_LENGTH2,
                   trainable=False)(sequence_input2)
wo_ind = IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                return_sequences=True)(wo_ind)
wo_ind = IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                return_sequences=True)(wo_ind)
wo_ind = IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                return_sequences=True)(wo_ind)
wo_ind = IndRNN(64, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                return_sequences=False)(wo_ind)
# wo_ind=Dense(200,activation='relu')(wo_ind)
print('woind', wo_ind)
wo_ind = Reshape((2, 32))(wo_ind)
print('woind', wo_ind)

sequence_input4 = Input(shape=(MAX_SEQUENCE_LENGTH4, ), dtype='int32')
te_cap = Embedding(len(embedding_matrix4),
                   EMBEDDING_DIM,
                   weights=[embedding_matrix4],
                   input_length=MAX_SEQUENCE_LENGTH4,
                   trainable=False)(sequence_input4)
from algorithm.capsule_layer import CategoryCap, PrimaryCap
from algorithm.attention_url import Attention_layer
k = 12
conv1 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1')(te_cap)
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
cap = BatchNormalization()(category_caps)
print('cap', cap)

# print('cap',cap)
# cap=Dense(200,activation='relu')(cap)
# cap = Flatten()(cap)

output = concatenate([wo_ind, cap])
print('out', output)
# output = Reshape((2*32))(output)
# print('out',output)
output = Attention_layer()(output)
print('out', output)

output = Dense(32, activation='relu')(output)
preds = Dense(1, activation='sigmoid')(output)
print("training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
model = Model(inputs=[sequence_input2, sequence_input4], outputs=[preds])
model.summary()

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])


def predict_url(urls):
    result = []
    _threshold = 0.5
    model.load_weights(save_model_file)
    text = []
    text4 = []
    for url in urls:
        tmp = []
        tmp4 = []
        for c in url:
            tmp.append(c)
        for c in url:
            tmp4.append(ord(c))
        text.append(tmp + tmp4)
        text4.append(tmp4)
    print(text)
    print(text4)
    se = tokenizer.texts_to_sequences(text)
    data = pad_sequences(se, maxlen=MAX_SEQUENCE_LENGTH2)
    print(data)
    se4 = tokenizer4.texts_to_sequences(text4)
    data4 = pad_sequences(se4, maxlen=MAX_SEQUENCE_LENGTH4)
    print(data4)
    pred_class = model.predict([data, data4]).reshape(len(urls))
    print(pred_class)
    for url, pred in zip(urls, pred_class):
        if pred > 0.5:
            result.append(url + '\t\t良性网站')
        else:
            result.append(url + '\t\t恶性网站')
    resultstr = '\n'.join(result)
    return resultstr


if __name__ == "__main__":
    urls = ['www.baidu.com', 'www.freebuf.com']
    print(predict_url(urls))