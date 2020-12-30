import os
import sys
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.FATAL)
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

# _vali_rate, batch_size, epochs, featuredim = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# print(_vali_rate, batch_size, epochs, featuredim, flush=True)


#词向量化
def train(_vali_rate, batch_size, epochs, featuredim):
    MAX_NB_WORDS = 0
    MAX_SEQUENCE_LENGTH2 = 115
    MAX_SEQUENCE_LENGTH4 = 70
    vali_rate = _vali_rate
    EMBEDDING_DIM = 200
    epoch = epochs
    T_fea1 = 'T6'
    T_fea2 = 'T4'
    T_model = 'our'
    save_model_file = 'models/' + T_fea1 + 'best_115' + T_model + '_train.h5'
    Read_data = './dataset_url/paper2_' + T_fea1 + '_data.csv'
    Dictionary_Path2 = './word2vec/paper2_' + T_fea1 + '_model_200.txt'
    Read_data4 = './dataset_url/paper2_' + T_fea2 + '_data.csv'
    Dictionary_Path4 = './word2vec/paper2_' + T_fea2 + '_model_200.txt'

    #########################################################设置GPU使用
    # import tensorflow as tf
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #use GPU with ID=0
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    # config.gpu_options.allow_growth = True  #allocate dynamically
    # sess = tf.Session(config=config)

    ####################################################################################T6数据读取
    #########################################################读入数据
    fdata = open(Read_data)  # 写入数据文件
    lines = fdata.readlines()  # 调用文件的 readline()方法
    texts2 = []
    labels = []
    for t in lines:
        t = t.strip().split(',')
        texts2.append(t[:-1])
        labels.append(int(t[-1]))
    #########################################################将数据URL处理 data:数据矩阵，labels标签
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters="", oov_token="unk")  #filters =‘’ 表示保留特殊字符
    tokenizer.fit_on_texts(texts2)
    sequences = tokenizer.texts_to_sequences(texts2)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index), flush=True)
    # data1 = pad_sequences(sequences)#, maxlen=MAX_SEQUENCE_LENGTH
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH2)  #, maxlen=MAX_SEQUENCE_LENGTH
    print(data)
    # labels2 = to_categorical(np.asarray(labels))
    labels = np.asarray(labels)

    #########################################################随机抽取训练集和验证集 x_train2，y_train2，x_val2，y_val2
    nb_validation_samples = int(vali_rate * data.shape[0])
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train2 = data[:-nb_validation_samples]
    y_train2 = labels[:-nb_validation_samples]
    x_val2 = data[-nb_validation_samples:]
    y_val2 = labels[-nb_validation_samples:]

    ####################################################################################T4数据读取
    fdata4 = open(Read_data4)  # 写入数据文件
    lines4 = fdata4.readlines()  # 调用文件的 readline()方法
    texts4 = []
    for t in lines4:
        t = t.strip().split(',')
        texts4.append(t[:-1])

    tokenizer4 = Tokenizer(num_words=MAX_NB_WORDS, filters="", oov_token="unk")  #filters =‘’ 表示保留特殊字符
    tokenizer4.fit_on_texts(texts4)
    sequences4 = tokenizer4.texts_to_sequences(texts4)
    word_index4 = tokenizer4.word_index
    print('Found %s unique tokens.' % len(word_index4), flush=True)
    # data1 = pad_sequences(sequences)#, maxlen=MAX_SEQUENCE_LENGTH
    data = pad_sequences(sequences4, maxlen=MAX_SEQUENCE_LENGTH4)  #, maxlen=MAX_SEQUENCE_LENGTH

    # 打乱顺序
    data4 = data[indices]

    x_train4 = data4[:-nb_validation_samples]
    x_val4 = data4[-nb_validation_samples:]

    #########################################################将字符嵌入向量读出来，对应的复制给embedding_matrix
    embeddings_index2 = {}
    f = open(Dictionary_Path2, 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index2[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index2), flush=True)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector2 = embeddings_index2.get(word)
        if embedding_vector2 is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector2
    print("embedding_matrix", embedding_matrix, flush=True)

    ###############################################################将字符嵌入向量读出来，对应的复制给embedding_matrix4
    embeddings_index4 = {}
    f4 = open(Dictionary_Path4, 'r', encoding='utf-8')
    for line4 in f4:
        values4 = line4.split()
        word4 = values4[0]
        coefs4 = np.asarray(values4[1:])
        embeddings_index4[word4] = coefs4
    f4.close()
    print('Found %s word vectors.' % len(embeddings_index4), flush=True)
    embedding_matrix4 = np.zeros((len(word_index4) + 1, EMBEDDING_DIM))

    for word4, i4 in word_index4.items():
        embedding_vector4 = embeddings_index4.get(word4)
        if embedding_vector4 is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix4[i4] = embedding_vector4
    print("embedding_matrix", embedding_matrix4, flush=True)

    ################################################################################模型部分
    from algorithm.indrnn import IndRNN
    sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH2, ), dtype='int32')
    wo_ind = Embedding(len(word_index) + 1,
                       EMBEDDING_DIM,
                       weights=[embedding_matrix],
                       input_length=MAX_SEQUENCE_LENGTH2,
                       trainable=False)(sequence_input2)
    wo_ind = IndRNN(128,
                    recurrent_clip_min=-1,
                    recurrent_clip_max=-1,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    return_sequences=True)(wo_ind)
    wo_ind = IndRNN(128,
                    recurrent_clip_min=-1,
                    recurrent_clip_max=-1,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    return_sequences=True)(wo_ind)
    wo_ind = IndRNN(128,
                    recurrent_clip_min=-1,
                    recurrent_clip_max=-1,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    return_sequences=True)(wo_ind)
    wo_ind = IndRNN(64,
                    recurrent_clip_min=-1,
                    recurrent_clip_max=-1,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    return_sequences=False)(wo_ind)
    # wo_ind=Dense(200,activation='relu')(wo_ind)
    print('woind', wo_ind)
    wo_ind = Reshape((2, 32))(wo_ind)
    print('woind', wo_ind)

    sequence_input4 = Input(shape=(MAX_SEQUENCE_LENGTH4, ), dtype='int32')
    te_cap = Embedding(len(word_index4) + 1,
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
    category_caps = CategoryCap(num_capsule=num_capsule1,
                                dim_vector=dim_capsule1,
                                num_routing=num_routing,
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

    ############################################################保存部分
    #损失不下降，提前结束
    # from keras.callbacks import EarlyStopping
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)#patience第10轮不下降，则提前停止
    from keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(
        filepath=save_model_file,  #(就是你准备存放最好模型的地方),
        monitor='val_acc',  #(或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
        verbose=1,  #(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
        save_best_only='True',  #(只保存最好的模型,也可以都保存),
        save_weights_only='True',
        mode='max',  #(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
        period=1)  #(checkpoints之间间隔的epoch数)
    #损失不下降，则自动降低学习率
    lrreduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.1,
                                                 patience=10,
                                                 verbose=0,
                                                 mode='auto',
                                                 epsilon=0.0001,
                                                 cooldown=0,
                                                 min_lr=0)

    import time
    fit_start = time.time()
    history = model.fit([x_train2, x_train4],
                        y_train2,
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=2,
                        shuffle=True,
                        validation_data=([x_val2, x_val4], y_val2),
                        callbacks=[checkpoint])
    fit_end = time.time()
    print("train time is: ", fit_end - fit_start)

    model.load_weights(save_model_file)
    t_start = time.time()
    scores = model.evaluate([x_val2, x_val4], y_val2, verbose=0)
    t_end = time.time()
    print('Test loss :', scores[0])
    print('Test accuracy :', scores[1])
    print("test time is: ", t_end - t_start)

    # y_pred_class = model.predict([x_val2]).reshape(len(y_val2))
    # #将y_val变为list类型             重要！！！！！！！！！！！！！！
    # y_val = np.array(y_val2)
    y_val = np.array(y_val2)
    print('x_val2', x_val2)
    print('x_val4', x_val4)
    print('y_val', y_val)
    print('y_val', type(y_val))
    y_pred_class = model.predict([x_val2, x_val4]).reshape(len(y_val2))
    #将y_val变为list类型             重要！！！！！！！！！！！！！！

    print('y_pred_class', y_pred_class)

    print('y_pred_class', type(y_pred_class))

    def threshold_y_val_pred(threshold, true_label, y_pred_class):
        pred_label = [int(item > threshold) for item in y_pred_class]
        from sklearn import metrics
        confusion = metrics.confusion_matrix(true_label, pred_label)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        Accuracy = metrics.accuracy_score(true_label, pred_label)
        Error = 1 - metrics.accuracy_score(true_label, pred_label)
        Precision = metrics.precision_score(true_label, pred_label)
        Recall = metrics.recall_score(true_label, pred_label)
        F1 = metrics.f1_score(true_label, pred_label)
        AUC = metrics.roc_auc_score(true_label, pred_label)
        print("TP:", TP, "TN:", TN, "TP:", FP, "FN:", FN, "\nAccuracy:", Accuracy, "Error:", Error, "\nPrecision:", Precision,
              "Recall:", Recall, "F1:", F1, "\nAUC:", AUC, "threshold:", threshold)
        # print("pred_label",pred_label)
        return TP, TN, FP, FN, Accuracy, Error, Precision, Recall, F1, AUC, threshold, pred_label

    TP, TN, FP, FN, Accuracy, Error, Precision, Recall, F1, AUC, thre, y_pre = threshold_y_val_pred(0.1, y_val, y_pred_class)
    threshold = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thr in range(len(threshold)):
        print("阈值", threshold[thr])
        TP2, TN2, FP2, FN2, Accuracy2, Error2, Precision2, Recall2, F12, AUC2, thre2, y_pre2 = threshold_y_val_pred(
            threshold[thr], y_val, y_pred_class)
        if F12 > F1:
            TP, TN, FP, FN, Accuracy, Error, Precision, Recall, F1, AUC, thre, y_pre = TP2, TN2, FP2, FN2, Accuracy2, Error2, Precision2, Recall2, F12, AUC2, thre2, y_pre2
    print("###################################################################")
    print("###################################################################")
    print("最大F值时，阈值是：", thre)
    print("TP:", TP, "TN:", TN, "TP:", FP, "FN:", FN, "\nAccuracy:", Accuracy, "Error:", Error, "\nPrecision:", Precision,
          "Recall:", Recall, "F1:", F1, "\nAUC:", AUC)


if __name__ == "__main__":
    os.chdir(r'D:\papers\恶意网站\项目\soft')
    sys.path.append(r'D:\papers\恶意网站\项目\soft')
    train(0.2, 128, 20, 200)