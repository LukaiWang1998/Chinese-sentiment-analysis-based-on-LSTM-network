#! /bin/env python
# -*- coding: utf-8 -*-
# encoding: utf-8

import pandas as pd
import numpy as np

from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence
import keras.utils
from keras import utils as np_utils
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool1D
from collections import Counter
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import keras.backend as K
import yaml
import sys
import multiprocessing

sys.setrecursionlimit(1000000)
reload(sys)
sys.setdefaultencoding('utf8')

np.random.seed()

#参数配置
cpu_count = multiprocessing.cpu_count()  # 4CPU数量
voc_dim = 300 #word的向量维度
min_out = 10 #单词出现次数
window_size = 7 #WordVec中的滑动窗口大小

lstm_input = 300 #lstm输入维度
epoch_time = 20 #epoch
batch_size = 32 #batch

#读取数据函数
def loadfile():
    #文件输入
    neg = []
    pos = []
    with open('../data/pos.txt', 'r') as f:
        for line in f.readlines():
            pos.append(line)
        f.close()
    with open('../data/neg.txt', 'r') as f:
        for line in f.readlines():
            neg.append(line)
        f.close()
        X_Vec = np.concatenate((pos, neg))
    y = np.concatenate((np.zeros(len(pos), dtype=int),
                        np.ones(len(neg), dtype=int)))
    # print X_Vec,y
    # print y
    return X_Vec, y,neg,pos

def loadfile1():
    data = pd.read_csv('../data/data_with_2label.csv')
    X = data['正文内容']
    y = data['标签']
    # print data['正文内容']
    # print data['标签']
    print Counter(data['标签'])
    return X, y

def loadfile2():
    data = pd.read_csv('../data/data_jg_with_2label.csv')
    X = data['正文内容']
    y = data['标签']
    # print data['正文内容']
    # print data['标签']
    print Counter(data['标签'])
    return X, y
#分词函数
def onecut(doc):
    # 将中文分成一个一个的字
    #print len(doc),ord(doc[0])
    #print doc[0]+doc[1]+doc[2]
    ret = [];
    i=0
    while i < len(doc):
        c=""
        #utf-8的编码格式，小于128的为1个字符，n个字符的化第一个字符的前n+1个字符是1110
        if ord(doc[i])>=128 and ord(doc[i])<192:
            print ord(doc[i])
            assert 1==0#所以其实这里是不应该到达的
            c = doc[i]+doc[i+1];
            i=i+2
            ret.append(c)
        elif ord(doc[i])>=192 and ord(doc[i])<224:
            c = doc[i] + doc[i + 1];
            i = i + 2
            ret.append(c)
        elif ord(doc[i])>=224 and ord(doc[i])<240:
            c = doc[i] + doc[i + 1] + doc[i + 2];
            i = i + 3
            ret.append(c)
        elif ord(doc[i])>=240 and ord(doc[i])<248:
            c = doc[i] + doc[i + 1] + doc[i + 2]+doc[i + 3];
            i = i + 4
            ret.append(c)
        else :
            assert ord(doc[i])<128
            while ord(doc[i])<128:
                c+=doc[i]
                i+=1
                if (i==len(doc)) :
                    break
                if doc[i] is " ":
                    break;
                elif doc[i] is ".":
                    break;
                elif doc[i] is ";":
                    break;
            ret.append(c)
    '''
    for i in range(len(ret)):
        print ret[i]
        if (i>=2):
            break;
    '''
    return ret

#对每条文档进行循环调用onecut分词函数
def one_seq(text):
    text1=[]
    for document in text:
        if len(document)<1:
            continue
        text1.append(onecut(document.replace('\n', '')) )
    return text1
#训练word2vec
def word2vec_train(X_Vec):
    model_word = Word2Vec(size=voc_dim,
                     min_count=min_out,
                     window=window_size,
                     workers=cpu_count,
                     iter=10)#word2vec参数设置
    model_word.build_vocab(X_Vec)
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.iter)#训练word2vec


    #print model_word.wv.vocab.keys()[54],model_word.wv.vocab.keys()
    #print len(model_word.wv.vocab.keys())
    # print model_word ['有']
    input_dim = len(model_word.wv.vocab.keys()) + 1 #下标0空出来给不够10的字
    print input_dim
    embedding_weights = np.zeros((input_dim, voc_dim)) #定义空权重数组
    w2dic={}
    for i in range(len(model_word.wv.vocab.keys())):
        embedding_weights[i+1, :] = model_word [model_word.wv.vocab.keys()[i]]
        w2dic[model_word.wv.vocab.keys()[i]]=i+1
    #print embedding_weights
    return input_dim,embedding_weights,w2dic

#将数据转成index
def data2inx(w2indx,X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    return data


class Metrics(keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision

        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))



def rec(y_true, y_pred,name='rec'):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def pre(y_true, y_pred,name='pre'):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def train_lstm(input_dim, embedding_weights, x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,
                        input_dim=input_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=lstm_input))
    model.add(Lambda(lambda x: x, output_shape=lambda s: s))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='SAME'))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2, strides=2)),
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='SAME'))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2, strides=2)),
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(256, activation='softsign',dropout=0.1))
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    print 'Compiling the Model...'
    model.compile(loss='binary_crossentropy',#hinge
                  optimizer='adam', metrics=['mae', 'acc',rec,pre])
    model.summary()

    print "Train..."  # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, verbose=1,validation_split=0.2,
              callbacks=[
                  # keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=1,),
                  keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max', ),
                  keras.callbacks.ModelCheckpoint('../model/posneg+唐山训练 jg测试.h5', monitor='val_acc',
                                                             verbose=1, save_best_only=True, mode='max', period=1),
                  # Metrics((x_valid, y_valid)),


              ]
              )

    print "Evaluate..."
    tangshan_model = load_model('../model/posneg+唐山训练 jg测试.h5',
                                compile=False)
    tangshan_model.compile(loss='binary_crossentropy',  # hinge
                  optimizer='adam', metrics=['mae', 'acc', rec, pre])
    score = tangshan_model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    y_pred = tangshan_model.predict(x_test,batch_size=batch_size)

    y_test = np.argmax(y_test,axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred,average='weighted')
    recall = recall_score(y_true=y_test, y_pred=y_pred,average='weighted')
    print 'accuracy:',accuracy
    print 'precision:', precision
    print 'recall:', recall



    yaml_string = model.to_yaml()
    # with open('../model/lstm.yml', 'w') as outfile:
    #     outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.save_weights('../model/lstm.h5')
    print 'Test score:', score


X_Vec1, y1,neg,pos = loadfile()#pos
print X_Vec1.shape
print y1.shape
X_Vec2, y2 = loadfile1()#tangshan
print X_Vec2.shape
print y2.shape
X_Vec3, y3 = loadfile2()#jg
print X_Vec3.shape
print y3.shape



X_Vec1 = one_seq(X_Vec1)
print len(X_Vec1)
X_Vec2 = one_seq(X_Vec2)
print len(X_Vec2)
X_Vec3 = one_seq(X_Vec3)
print len(X_Vec3)


input_dim_Vec3,embedding_weights_Vec3,w2dic_test_Vec3 = word2vec_train(X_Vec3)
index_Vec3 = data2inx(w2dic_test_Vec3,X_Vec3)
index2_Vec3 = sequence.pad_sequences(index_Vec3, maxlen=voc_dim)
X_Vec_train = np.concatenate((X_Vec1,X_Vec2))
input_dim_train,embedding_weights_train,w2dic_train = word2vec_train(X_Vec_train)
index_train = data2inx(w2dic_train,X_Vec_train )
index2_train = sequence.pad_sequences(index_train, maxlen=voc_dim )
index2_Vec3_test, index2_Vec3_valid, y3_test, y3_valid = train_test_split(index2_Vec3, y3, test_size=0.2,random_state=42)
x_train = np.concatenate((index2_train,index2_Vec3_test))
y_train = np.concatenate((y1,y2,y3_test))
x_test = index2_Vec3_valid
y_test = y3_valid
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)
embedding_weights = np.concatenate((embedding_weights_train,embedding_weights_Vec3))
input_dim =input_dim_train + input_dim_Vec3
train_lstm(input_dim, embedding_weights, x_train, y_train, x_test, y_test)