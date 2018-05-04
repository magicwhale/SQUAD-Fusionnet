from load import *

print("processing training data")
loadData('train-v1.1.json', 'training', 'data')
print("processing dev data")
loadData('dev-v1.1.json', 'dev', 'data')
print("processing embeddings")
processEmbeddings('data/glove.6B', 300, 'data') 
