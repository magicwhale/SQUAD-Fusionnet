#This code is loosely based on the code

import os
import json
import nltk
import pickle
from random import shuffle
from pprint import pprint
import numpy as np

PAD_TOK = b"<pad>"
UNK_TOK = b"<unk>"
SPECIAL_TOKS = [PAD_TOK, UNK_TOK]

def tokenize(string):
    return [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(string)]

def charToWordLoc(string, charNum):
    return len(tokenize(string[:charNum]))

def loadData(inFile, dataName, outDir):
    data = json.load(open(inFile))
    data = data['data']
    contextTokenList = []
    questionTokenList = []
    answerList = []
    spanList = []

    #Load Data into examples list
    for item in data:
        paragraphs = item['paragraphs']
        for par in paragraphs:
            #process context
            context = str(par['context']) 
            contextTokens = tokenize(context)

            # context = context.replace("''", '" ')
            # context = context.replace("``", '" ')
            # context = context.lower()
            qas = par['qas']

            for qa in qas:
                q = str(qa['question'])
                qTokens = tokenize(q)
                a = str(qa['answers'][0]['text']).lower()
                aStartChar = qa['answers'][0]['answer_start']
                aStart = charToWordLoc(context, aStartChar)
                aEnd = aStart + len(tokenize(a)) - 1

                contextTokenList.append(contextTokens)
                questionTokenList.append(qTokens)
                answerList.append(a)
                spanList.append([aStart, aEnd])

    #Write examples list to file
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    with open(os.path.join(outDir, dataName +'.context'), 'wb') as contextFile,  \
         open(os.path.join(outDir, dataName +'.question'), 'wb') as questionFile,  \
         open(os.path.join(outDir, dataName +'.answer'), 'wb') as answerFile,  \
         open(os.path.join(outDir, dataName +'.span'), 'wb') as spanFile:
        
        pickle.dump(contextTokenList, contextFile)
        pickle.dump(questionTokenList, questionFile)
        pickle.dump(answerList, answerFile)
        pickle.dump(spanList, spanFile)


def loadJsonData(jsonFile):
    with open(jsonFile) as dataFile:
        data = json.load(dataFile)
        return data


def extractCtxtQn(data):
    quesIdSet = []
    ctxtTokenSet = []
    quesTokenSet = []

    numEssays = len(data['data'])

    for essayIdx in range(numEssays):
        essayCtxts = data['data'][essayIdx]['paragraphs']
        for paraIdx in range(len(essayCtxts)):
            context = unicode(essayCtxts[paraIdx]['context'])
            ctxtTokens = tokenize(context)
            questions = essayCtxts[paraIdx]['qas']

            for q in questions:
                quesId = q['id']
                ques = unicode(q['question'])
                quesTokens = tokenize(ques)

                quesIdSet.append(quesId)
                ctxtTokenSet.append(ctxtTokens)
                quesTokenSet.append(quesTokens)

    return quesIdSet, ctxtTokenSet, quesTokenSet





                
def findAnswers(session, myModel, wordToId, contexts, questions):

    idToAns = {}  
    b = 0; # batch index  
    batches = generateBatches(wordToId, contexts, questions, spans, myModel.FLAGS.batch_size)

    for batch in batches:

    


def loadGlove(gloveDir, gloveDim):
    wordToId = {}
    idToWord = {}
    embMatrix = []
    currId = 0

    for token in SPECIAL_TOKS:
        wordToId[token] = currId
        idToWord[currId] = token
        embed = [np.random.randn() for _ in range(gloveDim)]
        embMatrix.append(embed)
        currId += 1


    with open(os.path.join(gloveDir, 'glove.6b.' + str(gloveDim) + 'd.txt'), 'r') as fp:
        for line in fp:
            tokens = line.split()
            word = tokens.pop(0)
            wordToId[word] = currId
            idToWord[currId] = word
            embed = [float(n) for n in tokens]
            embMatrix.append(embed)
            currId += 1
    
    embMatrix = np.array(embMatrix)
    return wordToId, idToWord, embMatrix


    #Make data into 

    '''
        should pad data, make mask, return list with context and question
        ids, masks, and tokens, as well as answer spans and tokens
    '''
# loadData('train-v1.1.json', 'training', 'data')
# w2i, i2w, mat = loadGlove("data/glove.6b", 50)
# print(mat.shape)
# print(w2i)
# print(i2w)
# pprint(loadData('train-v1.1.json'))