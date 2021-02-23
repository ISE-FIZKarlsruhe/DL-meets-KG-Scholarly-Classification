import pdb

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split



def splitTrainTest():
    #df = pd.read_csv('./datasets/AMiner/strict_cs_papers_5.csv')
    df = pd.read_csv('./datasets/ArXiv/arxiv-metadata-cs-balanced.csv')
    
    #TODO remove test set for unseen classes

    # split dataset
    trainX, testX, _, _ =train_test_split(range(len(df["label"])), df["label"], test_size=0.30)
    trainDf = df.iloc[trainX]
    testDf = df.iloc[testX]

    # write datasets
    #trainDf.to_csv("./datasets/AMiner/strict_cs_papers_5_train.csv", index = False)
    trainDf.to_csv("./datasets/ArXiv/arxiv-metadata-cs-balanced_train.csv", index = False)
    #testDf.to_csv("./datasets/AMiner/strict_cs_papers_5_test.csv", index = False)
    testDf.to_csv("./datasets/ArXiv/arxiv-metadata-cs-balanced_test.csv", index = False)

def filterArXivCS():
    
    df = pd.read_json("./datasets/ArXiv/arxiv-metadata-oai-snapshot.json", orient="records", lines=True)
    keywordsDf = pd.read_csv("./datasets/ArXiv/manualMappedKeywords.csv", sep=';', index_col="id")
    keywordsDf = keywordsDf[keywordsDf["group"] == "cs"]
    keywordSet = set(keywordsDf.index)
    
    df["categories"] = [x.split() for x in  df["categories"]]
    #df = df[[np.any([xx in keywordSet for xx in x]) for x in df["categories"]]]
    df = df[[np.all([xx in keywordSet for xx in x]) for x in df["categories"]]]
    #pdb.set_trace()

    df.set_index("id", inplace=True)
    #df.to_csv("./datasets/ArXiv/arxiv-metadata-cs-any.csv", sep=';')
    df.to_csv("./datasets/ArXiv/arxiv-metadata-cs-all.csv", sep=';')

def filterArXivCSlink():
    
    df = pd.read_json("./datasets/ArXiv/arxiv-metadata-oai-snapshot.json", orient="records", lines=True)
    keywordsDf = pd.read_csv("./datasets/ArXiv/manualMappedKeywords.csv", sep=';', index_col="id")
    keywordsDf = keywordsDf[keywordsDf["group"] == "cs"]
    pdb.set_trace()
    keywordsDf = keywordsDf.dropna(subset=["Wikidata", "DBpedia"], how="any")
    keywordSet = set(keywordsDf.index)
    
    df["categories"] = [x.split() for x in  df["categories"]]
    #df = df[[np.any([xx in keywordSet for xx in x]) for x in df["categories"]]]
    df = df[[np.all([xx in keywordSet for xx in x]) for x in df["categories"]]]
    #pdb.set_trace()

    df.set_index("id", inplace=True)
    #df.to_csv("./datasets/ArXiv/arxiv-metadata-cs-link-any.csv", sep=';')
    df.to_csv("./datasets/ArXiv/arxiv-metadata-cs-link-all.csv", sep=';')
def filterArXivCSEmbedding():

    df = pd.read_json("./datasets/ArXiv/arxiv-metadata-oai-snapshot.json", orient="records", lines=True)
    keywordsDf = pd.read_csv("./datasets/ArXiv/extendedKeywords.csv", sep=';', index_col="id")
    keywordsDf = keywordsDf[keywordsDf["group"] == "cs"]
    keywordsDf = keywordsDf.dropna(subset=["embedding"], how="any")
    keywordSet = set(keywordsDf.index)
    
    df["categories"] = [x.split() for x in  df["categories"]]
    #df = df[[np.any([xx in keywordSet for xx in x]) for x in df["categories"]]]
    df = df[[np.all([xx in keywordSet for xx in x]) for x in df["categories"]]]
    #pdb.set_trace()
    
    df.set_index("id", inplace=True)
    #df.to_csv("./datasets/ArXiv/arxiv-metadata-cs-link-any.csv", sep=';')
    df.to_csv("./datasets/ArXiv/arxiv-metadata-cs-embedding-all.csv", sep=';')

def createTrainTestSplit(documentDf):
    
    # split dataset
    trainX, testX, _, _ =train_test_split(range(len(df["label"])), df["label"], test_size=0.30)
    trainDf = documentDf.iloc[trainX]
    devDf = documentDf.iloc[devX]
    testDf = documentDf.iloc[testX]

    return trainDf, devDf, testDf 

def createZSLtrainTestSplit(documentDf):
    
    # generate test set for unseen classes 
    # TODO select unseen set automatically
    unseenSet = {"cs.CL", "cs.NI", "cs.DC", "cs.HC", "cs.SY", "cs.CG", "cs.MM", "cs.GR", "cs.SD", "cs.OS"} # every third class of sorted classes by documents --> 30% of classes, 27.1% of documents
    
    # TODO make key "categories" dynamic
    testValues = [np.any([xx in unseenSet for xx in x]) for x in  documentDf["categories"]]
    testUnseenDf =  documentDf[testValues]
    documentDf = documentDf[[ not x for x in testValues]]

    # generate test set for seen classes 
    shuffledIndex = list(documentDf.index)
    np.random.shuffle(shuffledIndex)
    testSeenDf = documentDf.loc[shuffledIndex[:int(0.3*len(shuffledIndex))]]
    documentDf = documentDf.loc[shuffledIndex[int(0.3*len(shuffledIndex)):]]

    # generate dev set
    shuffledIndex = list(documentDf.index)
    np.random.shuffle(shuffledIndex)
    devDf = documentDf.loc[shuffledIndex[:int(0.1*len(shuffledIndex))]]
    documentDf = documentDf.loc[shuffledIndex[int(0.1*len(shuffledIndex)):]]

    return documentDf, devDf, testUnseenDf, testSeenDf


def trainTestSplitArXiv():
    df = pd.read_csv("./datasets/ArXiv/arxiv-metadata-cs-embedding-all.csv", sep=';', index_col="id")
    df["categories"] = [ [xx.strip()[1:-1] for xx in x[1:-1].split(',')] for x in df["categories"]]

    # generate test set for unseen classes 
    unseenSet = {"cs.CL", "cs.NI", "cs.DC", "cs.HC", "cs.SY", "cs.CG", "cs.MM", "cs.GR", "cs.SD", "cs.OS"} # every third class of sorted classes by documents --> 30% of classes, 27.1% of documents
    testValues = [np.any([xx in unseenSet for xx in x]) for x in  df["categories"]]
    testUnseenDf =  df[testValues]
    df = df[[ not x for x in testValues]]

    # generate test set for seen classes 
    shuffledIndex = list(df.index)
    np.random.shuffle(shuffledIndex)
    testSeenDf = df.loc[shuffledIndex[:int(0.3*len(shuffledIndex))]]
    df = df.loc[shuffledIndex[int(0.3*len(shuffledIndex)):]]

    # generate dev set
    shuffledIndex = list(df.index)
    np.random.shuffle(shuffledIndex)
    devDf = df.loc[shuffledIndex[:int(0.1*len(shuffledIndex))]]
    df = df.loc[shuffledIndex[int(0.1*len(shuffledIndex)):]]

    testUnseenDf.to_csv("./datasets/ArXiv/test_unseen.csv", sep=';')
    testSeenDf.to_csv("./datasets/ArXiv/test_seen.csv", sep=';')
    devDf.to_csv("./datasets/ArXiv/dev.csv", sep=';')
    df.to_csv("./datasets/ArXiv/train.csv", sep=';')
    


def main():
    #splitTrainTest()
    #filterArXivCS()
    #filterArXivCSlink()
    #filterArXivCSEmbedding()
    trainTestSplitArXiv()

if __name__ == '__main__':
    main()
