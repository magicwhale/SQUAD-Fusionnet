import pickle
import os
from load import loadGlove
from batch import batch, generateBatches

def loadDataFiles(dataDir, dataName):
    with open(os.path.join(dataDir, dataName +'.context'), 'rb') as contextFile,  \
         open(os.path.join(dataDir, dataName +'.question'), 'rb') as questionFile,  \
         open(os.path.join(dataDir, dataName +'.answer'), 'rb') as answerFile,  \
         open(os.path.join(dataDir, dataName +'.span'), 'rb') as spanFile:

        contexts = pickle.load(contextFile)
        questions = pickle.load(questionFile)
        answers = pickle.load(answerFile)
        spans = pickle.load(spanFile)
        return contexts, questions, answers, spans

def main():
    # contexts, questions, answers, spans = loadDataFiles('data', 'training')
    wordToId, idToWord, embMat = loadGlove("data/glove.6b", 300)
    print(wordToId)
    # print(answers)
    # if FLAGS.mode == "train":
    #     with tf.Session() as sess:
    generateBatches(wordToId, contexts, questions, spans, 200)

if __name__ == "__main__":
    main()
    # tf.app.run()