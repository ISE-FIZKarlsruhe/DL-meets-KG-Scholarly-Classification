import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer 

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import hamming_loss
import gensim 

from dataless import DatalessModel

savePath = "./models/ArXiv/"

def preprocessTest(tokenizer, input_pair):
    documents = [x[0].lower() for x in input_pair]
    classes = [x[1].lower() for x in input_pair]

    document_sequences = tokenizer.texts_to_sequences(documents)
    class_sequences = tokenizer.texts_to_sequences(classes)

    documents_prep = [x.split() for x in tokenizer.sequences_to_texts(document_sequences)]
    class_prep = [x.split() for x in tokenizer.sequences_to_texts(class_sequences)]

    return documents_prep, class_prep

def test():
    # load dataset 
    #df = pd.read_csv("./datasets/AMiner/strict_cs_papers_5_test.csv")
    #df = pd.read_csv("./datasets/ArXiv/arxiv-metadata-cs-balanced_test.csv")
    df = pd.read_csv("./datasets/ArXiv/arxiv-metadata-cs-test-unbalanced.csv")
    keywordDf = pd.read_csv("./datasets/ArXiv/manualMappedKeywords.csv", sep=';', index_col="id")
    keywordDf = keywordDf[keywordDf["group"] == "cs"]
    df["keyword"] = [keywordDf["label"].loc[x] for x in df["keyword"]]
    
    df["doc"] = df["title"] + ' ' + df["abstracts"] # add title and abstracts together
    documents = list(df["doc"])

    keywords = list(df["keyword"])
    label = list(df["label"])
    #del df

    input_pair = [(x1, x2) for x1, x2 in zip(documents, keywords)]

    # load tokenizer
    with open(savePath + "tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    test_data_document, test_data_class = preprocessTest(tokenizer, input_pair)
    
    model = DatalessModel(0.5)
    vocab = gensim.models.KeyedVectors.load_word2vec_format('./datasets/GoogleNews-vectors-negative300.bin', binary=True)

    pred = model.predict([test_data_document, test_data_class], vocab)
    precision, recall, fscore, _ = precision_recall_fscore_support(label, pred, average="binary")
    print("precision: {:.3f} recall: {:.3f} fscore: {:.3f}".format(precision, recall, fscore))

    # compute hamming loss 

    #predDf = pd.DataFrame(data = {"doc":test_data_document, "class":test_data_class, "pred":pred})
    #predDf = predDf[predDf["pred"]==1]

    trueCategories = dict()
    for i,x in enumerate(label):
        if x == 1:
            if documents[i] in trueCategories:
                trueCategories[documents[i]].append(keywords[i])
            else:
                trueCategories[documents[i]] = [keywords[i]]
    predictedCategories = dict()
    for i,x in enumerate(pred):
        if x == 1:
            if documents[i] in predictedCategories:
                predictedCategories[documents[i]].append(keywords[i])
            else:
                predictedCategories[documents[i]] = [keywords[i]] 

    category_idx = {cat : i for i,cat in enumerate(keywordDf["label"])} 
    y_trueEncoded = []
    y_predEncoded = []
    for x in trueCategories.keys():
        y_true = trueCategories[x] if x in trueCategories else []
        y_pred = predictedCategories[x] if x in predictedCategories else [] 
        encTrue = [0] * len(keywordDf)
        for cat in y_true:
            idx = category_idx[cat]
            encTrue[idx] = 1
        y_trueEncoded.append(encTrue)
        encPred = [0] * len(keywordDf)
        for cat in y_pred:
            idx = category_idx[cat]
            encPred[idx] = 1
        y_predEncoded.append(encPred)
    
    print("hamming loss: {:.3f}".format(hamming_loss(y_trueEncoded,y_predEncoded)))

def main():
    test()

if __name__ == '__main__':
    main()
