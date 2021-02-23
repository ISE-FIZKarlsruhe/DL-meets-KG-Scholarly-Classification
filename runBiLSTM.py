from biLSTM import ZeroShotBiLSTM
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences 

import itertools
import pandas as pd
import numpy as np
import gensim
import random

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import hamming_loss

savePath = "./models/ArXiv/"

class Configuration(object):
	"""empty Configuration class"""
CONFIG = Configuration()
CONFIG.embedding_dim = 300
CONFIG.max_sequence_length_document = 215 # 80th percentile train
CONFIG.number_lstm_units_document = 50
CONFIG.rate_drop_lstm_document = 0.17
CONFIG.max_sequence_length_class =  5 # max label length train 
CONFIG.number_lstm_units_class = 50
CONFIG.rate_drop_lstm_class = 0.17
CONFIG.number_dense_units = 50
CONFIG.rate_drop_dense = 0.25
CONFIG.activation_function = "relu"
CONFIG.validation_split_ratio = 0.1
CONFIG.epochs = 200
CONFIG.batch_size = 256

def prepareDataset(docDf, keywordDf):
    negative_subsample =  5 
    binaryClassificationList = []
    
    # TODO check if usage of theoretical category set is useful
    keywordSet = set(itertools.chain(*docDf["categories"]))

    for i,doc in docDf.iterrows():
        
        #docData = doc["title"] + ". " +doc["abstract"]
        docData =  doc["abstract"]
        
        # positive examples
        for cat in doc["categories"]:
            classData = keywordDf["label"].loc[cat]
            binaryClassificationList.append(((docData,classData),1))
        # negative examples 
        for cat in random.sample(keywordSet - set(doc["categories"]), negative_subsample) :
            classData = keywordDf["label"].loc[cat]
            binaryClassificationList.append(((docData,classData),0))

    # shuffle data 
    random.shuffle(binaryClassificationList)
    # split data 
    input_pair, labels = zip(*binaryClassificationList)

    return input_pair, labels


def train():
    trainDf = pd.read_csv("./datasets/ArXiv/train.csv", sep=";", index_col="id")
    trainDf["categories"] = [[xx.strip()[1:-1] for xx in x[1:-1].split(",")] for x in trainDf["categories"]]
    devDf = pd.read_csv("./datasets/ArXiv/dev.csv", sep=';', index_col="id")
    devDf["categories"] = [[xx.strip()[1:-1] for xx in x[1:-1].split(",")] for x in devDf["categories"]]
    
    keywordDf = pd.read_csv("./datasets/ArXiv/extendedKeywords.csv", sep=';', index_col="id")
    trainKeywordSet = set(itertools.chain(*trainDf["categories"]))

    input_pair, label = prepareDataset(trainDf, keywordDf)
    input_pair_dev, label_dev = prepareDataset(devDf, keywordDf)
    del trainDf, devDf
    
    # load embeddings
    pretrainedWordVectors = gensim.models.KeyedVectors.load_word2vec_format('./datasets/GoogleNews-vectors-negative300.bin',binary=True) 
    # create tokenizer
    tokenizer = Tokenizer()
    texts = [x[0] for x in input_pair] + [x[1] for x in input_pair]
    tokenizer.fit_on_texts(texts)
    embedding_dim = CONFIG.embedding_dim   
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = pretrainedWordVectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            print("vector not found for word - %s" % word)
    
    # train
    model = ZeroShotBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length_document, CONFIG.number_lstm_units_document, CONFIG.rate_drop_lstm_document, CONFIG.max_sequence_length_class, CONFIG.number_lstm_units_class, CONFIG.rate_drop_lstm_class, CONFIG.number_dense_units, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio, CONFIG.epochs, CONFIG.batch_size)
    model_path = model.train_model(input_pair, label, input_pair_dev, label_dev, embedding_matrix, tokenizer, savePath)
    
    # save tokenizer
    with open(savePath + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepareDatasetTest(docDf, keywordDf, keywordList):
    
    binaryClassificationList = []

    for i,doc in docDf.iterrows():
        
        #docData = doc["title"] + ". " +doc["abstract"]
        docData =  doc["abstract"]
        
        for cat in keywordList:
            classData = keywordDf["label"].loc[cat]
            binaryClassificationList.append(((docData,classData),1 if cat in doc["categories"] else 0))
    # split data 
    input_pair, labels = zip(*binaryClassificationList)

    return input_pair, labels 


def test():
    unseenSet = {"cs.CL", "cs.NI", "cs.DC", "cs.HC", "cs.SY", "cs.CG", "cs.MM", "cs.GR", "cs.SD", "cs.OS" }

    # load datasets
    seenDf = pd.read_csv('./datasets/ArXiv/test.csv', sep=";", index_col="id")

    keywordDf = pd.read_csv("./datasets/ArXiv/extendedKeywords.csv", sep=';', index_col="id")
    #TODO filter keywords
    keywordDf = keywordDf.dropna(subset=["embeddingID"])
    keywordList = [x for x in keywordDf.index if x not in unseenSet]
    input_pair, label = prepareDatasetTest(seenDf, keywordDf, keywordList)
    
    # load tokenizer
    with open(savePath + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle) 
    # load model    
    model_path = savePath + "zeroShot_lstm.h5"  
    model = load_model(model_path)

    # Preprocessing
    # TODO change to object independent solution
    tmp_model = ZeroShotBiLSTM(CONFIG.embedding_dim, CONFIG.max_sequence_length_document, CONFIG.number_lstm_units_document, CONFIG.rate_drop_lstm_document, CONFIG.max_sequence_length_class, CONFIG.number_lstm_units_class, CONFIG.rate_drop_lstm_class, CONFIG.number_dense_units, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio, CONFIG.epochs, CONFIG.batch_size)
    test_data_document, test_data_class, test_label = tmp_model.preprocess(tokenizer, input_pair, label)

    # predict pairs
    predProp = list(model.predict([test_data_document, test_data_class], verbose=1).ravel())
    pred = [1 if x > 0.5 else 0 for x in predProp]
    
    # convert binary prediction 
    predMat = np.split(np.array(pred), len(keywordList))
    labelMat = np.split(np.array(label), len(keywordList))

    # Evaluation 
    precision, recall, fscore, _ = precision_recall_fscore_support(labelMat, predMat, average='micro')
    print("Micro) precision: {:.3f} recall: {:.3f} fscore: {:.3f}".format(precision, recall, fscore))
    precision, recall, fscore, _ = precision_recall_fscore_support(labelMat, predMat, average='macro')
    print("Macro) precision: {:.3f} recall: {:.3f} fscore: {:.3f}".format(precision, recall, fscore))
    print("hamming loss: {:.3f}".format(hamming_loss(labelMat, predMat)))

def main ():
    train()
    test()

if __name__ == '__main__':
    main()
