from random import shuffle
import numpy as np
import spacy

PAD_TOK = b"<pad>"
UNK_TOK = b"<unk>"

POS_DICT = {
    "PAD" : 0,
    "ADJ" : 1,
    "ADP" : 2,
    "ADV" : 3,
    "AUX" : 4,
    "CONJ" : 5,
    "CCONJ" : 6,
    "DET" : 7,
    "INTJ" : 8,
    "NOUN" : 9,
    "NUM" : 10,
    "PART" : 11,
    "PRON" : 12,
    "PROPN" : 13,
    "PUNCT": 14,
    "SCONJ": 15,
    "SYM" : 16,
    "VERB" : 17,
    "X" : 18,
    "SPACE" : 19
}


NER_DICT = {
    "PAD" : 0,
    "PERSON" : 1,
    "NORP" : 2,
    "FAC" : 3,
    "ORG" : 4,
    "GPE" : 5,
    "LOC" : 6,
    "PRODUCT" : 7,
    "EVENT" : 8,
    "WORK_OF_ART" : 9,
    "LAW" : 10,
    "LANGUAGE" : 11,
    "DATE" : 12,
    "TIME" : 13,
    "PERCENT": 14,
    "MONEY": 15,
    "QUANTITY" : 16,
    "ORDINAL" : 17,
    "CARDINAL" : 18,
    "": 19
}

class batch():
    def __init__(self, contextTokens, contextIds, contextPosIds, contextNerIds, contextFeatures, contextMask, qTokens, qIds, qMask, aTokens, aSpans, uuids=None, detokens=None):
        self.contextIds = contextIds
        self.contextPosIds = contextPosIds
        self.contextNerIds = contextNerIds
        self.contextMask = contextMask
        self.contextTokens = contextTokens
        self.contextFeatures = contextFeatures
        self.qIds = qIds
        self.qMask = qMask
        self.qTokens = qTokens
        self.aSpans = aSpans
        self.aTokens = aTokens
        self.uuids = uuids
        self.detokens = detokens
        self.batchSize = len(self.contextTokens)

    # def __init__(self, contextIds, contextMask, contextTokens, qIds, qMask, qTokens, aTokens):
    #     self.contextIds = contextIds
    #     self.contextMask = contextMask
    #     self.contextTokens = contextTokens
    #     self.qIds = qIds
    #     self.qMask = qMask
    #     self.qTokens = qTokens
    #     self.aTokens = aTokens


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

def padFeatures(featureBatch):
    maxLen = 0
    for features in featureBatch:
        maxLen = max(maxLen, len(features))
    padded = []
    for features in featureBatch:
        padded.append(features + [[0, 0, 0, 0]] * (maxLen - len(features)))
    return padded

def generateBatches(wordToId, contextData, questionData, spans, batchSize):
    contextTokens = contextData['tokens']
    contextIds = contextData['ids']
    contextPos = contextData['posIds']
    contextNer = contextData['nerIds']
    contextFeatures = contextData['features']

    qTokens = questionData['tokens']
    qIds = questionData['ids']

    batches = []
    shuffledIndices = np.arange(len(qTokens))
    np.random.shuffle(shuffledIndices)

    for batchStart in range(0, len(shuffledIndices), batchSize):
        batchCTokens = []
        batchCIds = []
        batchCPos = []
        batchCNer = []
        batchCFeatures = []

        batchQTokens = []
        batchQIds = []

        batchATokens = []
        batchASpans = []

        for n in range(batchStart, min(batchStart + batchSize, len(shuffledIndices))):
            i = shuffledIndices[n]

            cTokens = contextTokens[i]
            span = spans[i]

            batchCTokens.append(cTokens)
            batchCIds.append(contextIds[i])
            batchCPos.append(contextPos[i])
            batchCNer.append(contextNer[i])
            batchCFeatures.append(contextFeatures[i])

            batchQTokens.append(qTokens[i])
            batchQIds.append(qIds[i])

            batchATokens.append(cTokens[span[0]: span[1] + 1])
            batchASpans.append(span)

            # # Get POS tokens and convert them to ids
            # nlp = spacy.load('en_core_web_sm')
            # doc = spacy.tokens.doc.Doc(nlp.vocab, words=context, spaces=[True, False])
            # # run the standard pipeline against it
            # for name, proc in nlp.pipeline:
            #     doc = proc(doc)            
            # print(doc.text)

            # for token in doc:
            #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            #           token.shape_, token.is_alpha, token.is_stop)
            #     contextPosIds.append(POS_DICT[token.pos_])

            # # NER
            # for ent in doc.ents:
            #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
            #     contextNERids.append(NER_DICT[ent.label_])    
            
            # qTokens.append(question)
            # qIds.append(tokensToIds(wordToId, question))

            # # print(context)
            # # print(span)
            # aTokens.append(context[span[0] : span[1] + 1])
            # aSpans.append(span)

        #pad and get masks
        padId = wordToId[PAD_TOK]
        batchCIds, contextMask = pad(padId, batchCIds)
        batchCPos, _ = pad(POS_DICT["PAD"], batchCPos)
        batchCNer, _ = pad(NER_DICT["PAD"], batchCNer)
        batchCFeatures = padFeatures(batchCFeatures)
        batchQIds, qMask = pad(padId, batchQIds)

        #convert everything to np array
        batchCIds = np.array(batchCIds)
        batchCPos = np.array(batchCPos)
        batchCNer = np.array(batchCNer)
        batchCFeatures = np.array(batchCFeatures).astype(float)
        batchQIds = np.array(batchQIds)
        batchASpans = np.array(batchASpans)

        print(batchCFeatures.shape)

        newBatch = batch(batchCTokens, batchCIds, batchCPos, batchCNer, batchCFeatures, contextMask, batchQTokens, batchQIds, qMask, batchATokens, batchASpans)
        batches.append(newBatch)

    return batches


def generateBatches2(wordToId, contextData, questionData, batchSize):
    contextTokens = contextData['tokens']
    contextDetokens = contextData['detokens']
    contextIds = contextData['ids']
    contextPos = contextData['posIds']
    contextNer = contextData['nerIds']
    contextFeatures = contextData['features']

    qTokens = questionData['tokens']
    qIds = questionData['ids']
    qUniqueIds = questionData['uniqueIds']

    batches = []
    shuffledIndices = np.arange(len(qTokens))
    np.random.shuffle(shuffledIndices)

    for batchStart in range(0, len(shuffledIndices), batchSize):
        batchCTokens = []
        batchCDetokens = []
        batchCIds = []
        batchCPos = []
        batchCNer = []
        batchCFeatures = []

        batchQTokens = []
        batchQIds = []
        batchQuniqueIds = []


        for n in range(batchStart, min(batchStart + batchSize, len(shuffledIndices))):
            i = shuffledIndices[n]

            cTokens = contextTokens[i]

            batchCTokens.append(cTokens)
            batchCDetokens.append(contextDetokens[i])
            batchCIds.append(contextIds[i])
            batchCPos.append(contextPos[i])
            batchCNer.append(contextNer[i])
            batchCFeatures.append(contextFeatures[i])

            batchQTokens.append(qTokens[i])
            batchQIds.append(qIds[i])
            batchQuniqueIds.append(qUniqueIds[i])


        #pad and get masks
        padId = wordToId[PAD_TOK]
        batchCIds, contextMask = pad(padId, batchCIds)
        batchCPos, _ = pad(POS_DICT["PAD"], batchCPos)
        batchCNer, _ = pad(NER_DICT["PAD"], batchCNer)
        batchCFeatures = padFeatures(batchCFeatures)
        batchQIds, qMask = pad(padId, batchQIds)

        #convert everything to np array
        batchCIds = np.array(batchCIds)
        batchCPos = np.array(batchCPos)
        batchCNer = np.array(batchCNer)
        batchCFeatures = np.array(batchCFeatures).astype(float)
        batchQIds = np.array(batchQIds)

        print(batchCFeatures.shape)

        newBatch = batch(batchCTokens, 
            batchCIds, 
            batchCPos, 
            batchCNer, 
            batchCFeatures, 
            contextMask, 
            batchQTokens, 
            batchQIds, 
            qMask, 
            None, 
            None, 
            uuids=batchQuniqueIds,
            detokens=batchCDetokens)

        batches.append(newBatch)

    return batches


# # This is the same as generateBatches but with no references to span
# def generateBatches2(wordToId, uuids, contexts, questions, batchSize):
#     batches = []
#     shuffledIndices = np.arange(len(contexts))
#     np.random.shuffle(shuffledIndices)

#     for batchStart in range(0, len(shuffledIndices), batchSize):
#         contextTokens = []
#         contextIds = []
#         contextPosIds = []
#         contextNERids = []
#         qTokens = []
#         qIds = []
#         aTokens = []
#         batchUuids = []

#         for n in range(batchStart, min(batchStart + batchSize, len(shuffledIndices))):
#             i = shuffledIndices[n]
#             context = contexts[i]
#             question = questions[i]
#             uuid = uuids[i]

#             #Get ids from words
#             contextTokens.append(context)
#             contextIds.append(tokensToIds(wordToId, context))

#             # Get pos tokens and convert them to ids
#             nlp = spacy.load('en_core_web_sm')             
#             doc = spacy.tokens.doc.Doc(nlp.vocab, words=context, spaces=[True, False])
#             # run the standard pipeline against it
#             for name, proc in nlp.pipeline:
#                 doc = proc(doc)  
#             print(doc.text)
#             for token in doc:
#                 print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#                       token.shape_, token.is_alpha, token.is_stop)
#                 posTokens.append(token.pos_)
#                 contextPosIds.append(POS_DICT[token.pos_])

#             # NER
#             for ent in doc.ents:
#                 print(ent.text, ent.start_char, ent.end_char, ent.label_)
#                 contextNERids.append(NER_DICT[ent.label_])   

#             qTokens.append(question)
#             qIds.append(tokensToIds(wordToId, question))

#             batchUuids.append(uuid)

#         #pad and get masks
#         padId = wordToId[PAD_TOK]
#         contextIds, contextMask = pad(padId, contextIds)
#         contextPosIds, contextMask = pad(POS_DICT["PAD"], contextPosIds)
#         contextNERids, contextMask = pad(NER_DICT["PAD"], contextNERids)
#         qIds, qMask = pad(padId, qIds)

#         #convert everything to np array
#         contextIds = np.array(contextIds)
#         contextPosIds = np.array(contextPosIds)
#         contextNERids = np.array(contextNERids)
#         qIds = np.array(qIds)
#         # aSpans = np.array(aSpans)

#         newBatch = batch(contextIds, contextPosIds, contextNERids, contextMask, contextTokens, qIds, qMask, qTokens, None, aTokens, uuids=batchUuids)
#         batches.append(newBatch)

#     return batches