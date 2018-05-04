from load import *

print("processing embeddings")
w2i, _, _, _ = processEmbeddings('data/glove.6B', 300, 'data') 
print("processing training data")
loadData(w2i, 'train-v1.1.json', 'training', 'data')
print("processing dev data")
loadData(w2i, 'dev-v1.1.json', 'dev', 'data')
