from random import shuffle
import numpy as np
from load import UNK_TOK, PAD_TOK

class batch():
    def __init__(self, contextIds, contextMask, contextTokens, qIds, qMask, qTokens, aSpans, aTokens):
        self.contextIds = contextIds
        self.contextMask = contextMask
        self.contextTokens = contextTokens
        self.qIds = qIds
        self.qMask = qMask
        self.qTokens = qTokens
        self.aSpans = aSpans
        self.aTokens = aTokens
        self.batchSize = len(self.contextTokens)

def tokensToIds(wordToId, tokens):
    unkId = wordToId[UNK_TOK]
    return [wordToId.get(w, unkId) for w in tokens]

def pad(padId, tokenBatch):
    maxLen = 0
    for tokens in tokenBatch:
        maxLen = max(maxLen, len(tokens))
    padded = []
    mask = []
    for tokens in tokenBatch:
        padded.append(tokens + [padId] * (maxLen - len(tokens)))
        mask.append([1] * len(tokens) + [0] * (maxLen - len(tokens)))
    return padded, mask

def generateBatches(wordToId, contexts, questions, spans, batchSize):
    batches = []
    shuffledIndices = np.arange(len(contexts))
    np.random.shuffle(shuffledIndices)

    for batchStart in range(0, len(shuffledIndices), batchSize):
        contextTokens = []
        contextIds = []
        qTokens = []
        qIds = []
        aTokens = []
        aSpans = []

        for n in range(batchStart, min(batchStart + batchSize, len(shuffledIndices))):
            i = shuffledIndices[n]
            context = contexts[i]
            question = questions[i]
            span = spans[i]

            #Get ids from words
            contextTokens.append(context)
            contextIds.append(tokensToIds(wordToId, context))

            qTokens.append(question)
            qIds.append(tokensToIds(wordToId, question))

            aTokens.append(context[span[0]:span[1] + 1])
            aSpans.append(span)

        #pad and get masks
        padId = wordToId[PAD_TOK]
        contextIds, contextMask = pad(padId, contextIds)
        qIds, qMask = pad(padId, qIds)

        #convert everything to np array
        contextIds = np.array(contextIds)
        qIds = np.array(qIds)
        aSpans = np.array(aSpans)

        newBatch = batch(contextIds, contextMask, contextTokens, qIds, qMask, qTokens, aSpans, aTokens)
        batches.append(newBatch)

    return batches