#This code is loosely based on the code

import os
import json
# import nltk
# from nltk.tokenize.moses import MosesDetokenizer
import spacy
import pickle
from random import shuffle
from pprint import pprint
import numpy as np
from batch import generateBatches2
from batch import UNK_TOK, PAD_TOK
from batch import POS_DICT, NER_DICT
from keras.models import load_model
from collections import Counter

SPECIAL_TOKS = [PAD_TOK, UNK_TOK]

# def tokensToInput(tokens, gloveDict):
#     gloveEmb = [[] for line in ]

#     # COVE id 
#     coveEmbeddings = cove_model.predict(tokens)

#     # GLOVE id


#     # POS id

#     # NER id

#     # TF

#     # Question-context matching


# def tokenBatchToInput(batch, gloveDict):
#     gloveEmbs = [[gloveDict[token] for token in line] for line in batch ]

#     coveModel = load_model('CoVe/Keras_CoVe.h5')
#     coveEmbeddings = coveModel.

nlp = spacy.load('en_core_web_sm')

def tokensToIds(wordToId, tokens):
    unkId = wordToId[UNK_TOK]
    return [wordToId.get(w, unkId) for w in tokens]

def tokenize(string):
    return [token.text.replace("``", '"').replace("''", '"') for token in nlp(string)]

def toDoc(string):
    return nlp(string.replace("``", '"').replace("''", '"'))

def charToWordLoc(string, charNum):
    return len(tokenize(string[:charNum]))

def getFeatures(contextDoc, qDoc):
    #Calculate TF
    cnt = Counter()
    for token in contextDoc:
        cnt[token.text.lower()] += 1
    numTokens = sum(cnt.values())
    # tf = [cnt[token.text.lower()] / numTokens for token in contextDoc]

    # Get EM features
    qWords = {token.text for token in qDoc}
    qLower = {word.lower() for word in qWords}
    qLemma = {token.lemma_ if token.lemma_ != '-PRON-' else token.text.lower() for token in qDoc}

    features = []
    for token in contextDoc:
        tf = cnt[token.text.lower()] / numTokens
        matchWord = token.text in qWords
        matchLower = token.text.lower() in qLower
        matchLemma = (token.lemma_ if token.lemma_ != '-PRON-' else token.text.lower()) in qLemma
        features.append([tf, matchWord, matchLower, matchLemma])

    return features


def loadData(wordToId, inFile, dataName, outDir):
    nlp = spacy.load('en_core_web_sm')

    data = json.load(open(inFile))
    data = data['data']
    contextTokens = []
    contextIds = []
    contextPosIds = []
    contextNerIds = []
    contextFeatures = []
    qTokens = []
    qIds = []
    aTokens = []
    aSpans = []

    #Load Data into examples list
    for item in data:
        paragraphs = item['paragraphs']
        for par in paragraphs:
            # process context
            context = str(par['context']) 
            contextDoc = toDoc(context)
            cDocTokens = [token.text.lower() for token in contextDoc]
            cDocIds = tokensToIds(wordToId, cDocTokens)
            cDocPosIds = [POS_DICT[token.pos_] for token in contextDoc]
            cDocNerIds = [NER_DICT[token.ent_type_] for token in contextDoc]

            # context = context.replace("''", '" ')
            # context = context.replace("``", '" ')
            # context = context.lower()
            qas = par['qas']

            for qa in qas:
                q = str(qa['question'])
                qDoc = toDoc(q)
                qDocTokens = [token.text.lower() for token in qDoc]
                qTokens.append(qDocTokens)
                qIds.append(tokensToIds(wordToId, qDocTokens))

                # process answer
                a = str(qa['answers'][0]['text']).lower()
                aStartChar = qa['answers'][0]['answer_start']
                aStart = charToWordLoc(context, aStartChar)
                aEnd = aStart + len(tokenize(a)) - 1
                aTokens.append(cDocTokens[aStart : aEnd + 1])
                aSpans.append([aStart, aEnd])

                contextTokens.append(cDocTokens)
                contextIds.append(cDocIds)

                contextPosIds.append(cDocPosIds)
                contextNerIds.append(cDocNerIds)
                contextFeatures.append(getFeatures(contextDoc, qDoc))

    contextData = {
        'tokens': contextTokens,
        'ids': contextIds,
        'posIds': contextPosIds,
        'nerIds': contextNerIds,
        'features': contextFeatures
    }

    questionData = {
        'tokens': qTokens,
        'ids': qIds
    }

    #Write examples list to file
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    with open(os.path.join(outDir, dataName +'.context'), 'wb') as contextFile,  \
         open(os.path.join(outDir, dataName +'.question'), 'wb') as questionFile,  \
         open(os.path.join(outDir, dataName +'.answer'), 'wb') as answerFile,  \
         open(os.path.join(outDir, dataName +'.span'), 'wb') as spanFile:
        
        pickle.dump(contextData, contextFile)
        pickle.dump(questionData, questionFile)
        pickle.dump(aTokens, answerFile)
        pickle.dump(aSpans, spanFile)


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
            context = str(essayCtxts[paraIdx]['context'])
            ctxtTokens = tokenize(context)
            questions = essayCtxts[paraIdx]['qas']

            for q in questions:
                quesId = q['id']
                ques = str(q['question'])
                quesTokens = tokenize(ques)

                quesIdSet.append(quesId)
                ctxtTokenSet.append(ctxtTokens)
                quesTokenSet.append(quesTokens)

    return quesIdSet, ctxtTokenSet, quesTokenSet

                
def findAnswers(session, myModel, wordToId, quesIdSet, contexts, questions):

    idToAns = {}  
    b = 0; # batch index  
    detokenizer = MosesDetokenizer()

    batches = generateBatches2(wordToId, quesIdSet, contexts, questions, myModel.FLAGS.batch_size)

    for batch in batches:
        startBatch, endBatch = myModel.getSpans(session, batch)
        startBatch = startBatch.tolist()
        endBatch = endBatch.tolist()

        for e, (start, end) in enumerate(zip(startBatch, endBatch)):
            contextTokens = batch.contextTokens[e]
            answerTokens = contextTokens[start:end+1]

            uniqueId = batch.uuids[e]
            idToAns[uniqueId] = detokenizer.detokenize(answerTokens, return_str=True)
        b += 1

    return idToAns


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


    with open(os.path.join(gloveDir, 'glove.6B.' + str(gloveDim) + 'd.txt'), 'r') as fp:
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

def processEmbeddings(gloveDir, gloveDim, outDir):
    w2i, i2w, gloveMat = loadGlove(gloveDir, gloveDim)

    coveModel = load_model('CoVe/Keras_CoVe.h5')
    coveEmbs = coveModel.predict(np.expand_dims(gloveMat, axis=0))
    coveMat = np.squeeze(coveEmbs)

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    with open(os.path.join(outDir, 'glove.w2i'), 'wb') as w2i_file,  \
         open(os.path.join(outDir, 'glove.i2w'), 'wb') as i2w_file,  \
         open(os.path.join(outDir, 'glove.embMat'), 'wb') as gloveMat_file, \
         open(os.path.join(outDir, 'cove.embMat'), 'wb') as coveMat_file:
        
        pickle.dump(w2i, w2i_file)
        pickle.dump(i2w, i2w_file)
        pickle.dump(gloveMat, gloveMat_file)
        pickle.dump(coveMat, coveMat_file)

# def loadGloveAndCove():
#     wordToId = {}
#     idToWord = {}
#     gloveMat = []
#     coveMat = []
#     currId = 0

#     for token in SPECIAL_TOKS:
#         wordToId[token] = currId
#         idToWord[currId] = token
#         embed = [np.random.randn() for _ in range(gloveDim)]
#         gloveMat.append(embed)
#         currId += 1


#     with open(os.path.join(gloveDir, 'glove.6B.' + str(gloveDim) + 'd.txt'), 'r') as fp:
#         for line in fp:
#             tokens = line.split()
#             word = tokens.pop(0)
#             wordToId[word] = currId
#             idToWord[currId] = word
#             embed = [float(n) for n in tokens]
#             gloveMat.append(embed)
#             currId += 1
    
#     gloveMat = np.array(gloveMat)
#     return wordToId, idToWord, gloveMat
    # Make Cove embedding matrix

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
