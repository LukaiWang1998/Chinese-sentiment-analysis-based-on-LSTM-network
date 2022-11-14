# Chinese-sentiment-analysis-based-on-LSTM-network

## Background

Sentiment analysis is a typical task in Natural Language Processing (NLP). This model is aimed to build a model based on LSTM neural network to achieve sentiment analysis of the Chinese social media reviews on crime trials.

## Data preprocessing

Three datasets are used. A corpus which contains about 8000 positive and 8000 negative reviews is used as part of the training set. Around 7000 reviews on the trial of Tangshan Attack and around 23000 reviews on the trial of Jiangge case were obtained with web crawler and used as either the training set or the test set in training the model. The preprocessing process includes labelling, tokenization and word embedding.

### Labelling

The reviews of Tangshan attack and Jiangge case were labelled as positive or negative using Python library SnowNLP. Reviews with unclear emotional polarities were discarded during this process.

### Tokenization

The most common tokenization method in Chinese text processing is jieba. However, the test is directly segmented by characters instead of using jieba due to the limited size of the data. UTF-8 encoding is applied here for segmentation.

### Segmentation by characters

```
def onecut(doc):
    #print len(doc),ord(doc[0])
    #print doc[0]+doc[1]+doc[2]
    ret = [];
    i=0
    while i < len(doc):
        c=""
        #print i,ord(doc[i])
        if ord(doc[i])>=128 and ord(doc[i])<192:
            print ord(doc[i])
            assert 1==0
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
    return ret
```

## Word embedding

Now the segmented texts need to be vectorized, which is also called word embedding. Word embedding transforms text into a form that is compatible with a neural network. In this experiment, word embedding was performed using Python library Word2Vec to represent words as high-dimensional vectors. The distance between two words in a vector space indicates their semantic or contextual similarity.

```
def word2vec_train(X_Vec):
    model_word = Word2Vec(size=voc_dim,
                     min_count=min_out,
                     window=window_size,
                     workers=cpu_count,
                     iter=5)
    model_word.build_vocab(X_Vec)
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.iter)
    model_word.save('../model/Word2vec_model.pkl')

    input_dim = len(model_word.wv.vocab.keys()) + 1 
    embedding_weights = np.zeros((input_dim, voc_dim)) 
    w2dic={}
    for i in range(len(model_word.wv.vocab.keys())):
        embedding_weights[i+1, :] = model_word [model_word.wv.vocab.keys()[i]]
        w2dic[model_word.wv.vocab.keys()[i]]=i+1
    return input_dim,embedding_weights,w2dic
```

## Training process

### Activation function

Softsign is used as the activation function since it is more suitable for LSTM than tanh.

### Loss function

Three different loss functions :mse，hinge and binary_crossentropy were considered in this model. Eventually binary_crossentropy was applied since it was designed for binary classification. 

### Evaluation critera

The accuracy (acc) and the mean absolute error (mae) were used as evaluation criteria in training the model.

## Results

Take posneg+唐山训练 jg测试.py (The corpus and reviews on Tangshan Attack are used as training and validation set and the reviews on Jiangge case are used as the test set) as an example, the accuracy reached around 92%:

![](https://github.com/LukaiWang1998/Chinese-sentiment-analysis-based-on-LSTM-network/blob/main/data/jg_test_result.png)
