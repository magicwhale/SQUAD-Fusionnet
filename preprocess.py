from load import *

loadData('train-v1.1.json', 'training', 'data')
loadData('dev-v1.1.json', 'dev', 'data')
processGlove('data/glove.6B', 300, 'data') 
