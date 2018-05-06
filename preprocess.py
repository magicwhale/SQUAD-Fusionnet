from load import *
from main import loadEmbedFiles

print("processing embeddings")
processEmbeddings('data/glove.6B', 300, 'data') 

wordToId, idToWord, gloveMat, coveMat = loadEmbedFiles('data')

print("processing training data")
loadData(wordToId, 'train-v1.1.json', 'training', 'data')

print("processing dev data")
loadData2(wordToId, 'dev-v1.1.json', 'dev', 'data')
