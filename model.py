import tensorflow as tf
import os
# os.environ["KERAS_BACKEND"] = "theano"
# import keras as ks
import logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from graphComponents import *
from batch import *
from evaluate import f1_score
from batch import POS_DICT, NER_DICT
# from keras import backend as K

NUM_POS_TAGS = len(POS_DICT)
POS_EMB_SIZE = 12
NUM_NER_TAGS = len(NER_DICT)
NER_EMB_SIZE = 8

class Model():

    def __init__(self, FLAGS, wordToId, idToWord, gloveMat, coveMat):
        self.FLAGS = FLAGS
        self.wordToId = wordToId
        self.idToWord = idToWord

        # Build everything related to the graph
        with tf.variable_scope("model", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.initPlaceholders()
            self.addEmbedding(gloveMat, coveMat)
            self.addGraph()
            self.addLoss()

        # Training variables
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradientNorm = tf.global_norm(gradients)
        clippedGradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.paramNorm = tf.global_norm(params)

        self.globalStep = tf.Variable(0, name="globalStep", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.updates = opt.apply_gradients(zip(clippedGradients, params), global_step=self.globalStep)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestSaver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()

    def initPlaceholders(self):
        #Context, question, and answer placeholders
        # self.contextLen = tf.placeholder(tf.int32, shape=())
        # self.qLen = tf.placeholder(tf.int32, shape=())
        self.contextIds = tf.placeholder(tf.int32, shape=[None, None])
        self.contextPosIds = tf.placeholder(tf.int32, shape=[None, None])
        self.contextNerIds = tf.placeholder(tf.int32, shape=[None, None])
        self.contextFeatures = tf.placeholder(tf.float32, shape=[None, None])
        self.contextMask = tf.placeholder(tf.int32, shape=[None, None])

        self.qIds = tf.placeholder(tf.int32, shape=[None, None])
        self.qMask = tf.placeholder(tf.int32, shape=[None, None])
        self.aSpans = tf.placeholder(tf.int32, shape=[None, 2])

        #Dropout placeholder
        self.keepProb = tf.placeholder_with_default(1.0, shape=())

    def addEmbedding(self, gloveMat, coveMat):
        with vs.variable_scope("embedding"):

            gloveEmbMatrix = tf.constant(gloveMat, dtype=tf.float32, name="gloveMat")
            self.contextGlove = embedding_ops.embedding_lookup(gloveEmbMatrix, self.contextIds)
            self.qGlove = embedding_ops.embedding_lookup(gloveEmbMatrix, self.qIds)

            coveEmbMatrix = tf.constant(coveMat, dtype=tf.float32, name="coveMat")
            self.contextCove = embedding_ops.embedding_lookup(coveEmbMatrix, self.contextIds)
            self.qCove = embedding_ops.embedding_lookup(coveEmbMatrix, self.qIds)

            #21 POS tags
            posEmbMatrix = tf.get_variable("posEmbeddings", [NUM_POS_TAGS, POS_EMB_SIZE])
            self.contextPos = embedding_ops.embedding_lookup(posEmbMatrix, self.contextPosIds)

            nerEmbMatrix = tf.get_variable("nerEmbeddings", [NUM_NER_TAGS, NER_EMB_SIZE])
            self.contextNer = embedding_ops.embedding_lookup(nerEmbMatrix, self.contextNerIds)


    def addGraph(self):

        # Glove self attention
        gloveSelfAtt = wordLevelFusion(self.contextGlove, self.qGlove, "selfAttGlove")

        contextInput = tf.concat([self.contextGlove, self.contextCove, self.contextPos, self.contextNer, gloveSelfAtt], axis=-1)

        # High and low level representation
        lowLevelC = biLSTM(contextInput, self.FLAGS.hidden_size, self.keepProb, "lowLevelC",mask=self.contextMask)
        # TODO: modify so that hidden size can be different between levels
        highLevelC = biLSTM(lowLevelC, self.FLAGS.hidden_size, self.keepProb, "highLevelC", mask=self.contextMask)
        lowLevelQ = biLSTM(self.qGlove, self.FLAGS.hidden_size, self.keepProb, "lowLevelQ", mask=self.qMask)
        highLevelQ = biLSTM(lowLevelQ, self.FLAGS.hidden_size, self.keepProb, "highLevelQ", mask=self.qMask)

        # Question Understanding
        qUnderstanding = biLSTM(tf.concat([lowLevelQ, highLevelQ], axis=-1), self.FLAGS.hidden_size, self.keepProb, "qUnderstanding", mask=self.qMask)

        # Create history of word
        contextHOW = tf.concat([self.contextGlove, self.contextCove, lowLevelC, highLevelC], axis=-1)
        qHOW = tf.concat([self.qGlove, self.qCove, lowLevelQ, highLevelQ], axis=-1)

        # Fully-Aware Multi-level Fusion: Higher-level
        lowLevelFusion = fusion(contextHOW, qHOW, lowLevelQ, self.FLAGS.fusion_size, self.keepProb, "lowLevelFusion")
        highLevelFusion = fusion(contextHOW, qHOW, highLevelQ, self.FLAGS.fusion_size, self.keepProb, "highLevelFusion")
        understandingFusion = fusion(contextHOW, qHOW, qUnderstanding, self.FLAGS.fusion_size, self.keepProb, "understandingFusion")

        fusionContextHOW = tf.concat([
            lowLevelC,
            highLevelC,
            lowLevelFusion,
            highLevelFusion,
            understandingFusion],
            axis=-1)
        cqFusedRep = biLSTM(fusionContextHOW, self.FLAGS.hidden_size, self.keepProb, "cqFusedRep", mask=self.contextMask)

        # Fully-Aware Self-Boosted Fusion
        selfBoostedHOW = tf.concat([
            contextHOW,
            lowLevelFusion,
            highLevelFusion,
            understandingFusion,
            cqFusedRep],
            axis=-1)
        selfBoostedFusion = fusion(selfBoostedHOW, selfBoostedHOW, cqFusedRep, self.FLAGS.fusion_size, self.keepProb, "selfBoostedFusion")
        cUnderstanding = biLSTM(tf.concat([cqFusedRep, selfBoostedFusion], axis=-1),
            self.FLAGS.hidden_size, self.keepProb, "finalContext", mask=self.contextMask)
        # Get summarized question
        # with tf.variable_scope("questionSummary"):
        #   W = tf.get_variable("W", shape=(qUnderstanding.get_shape()[-1], 1), dtype=tf.float32)
        #   uqW = multiplyBatch(qUnderstanding, W)
        #   weights = tf.nn.softmax(uqW, axis=1)
        #   qSummary = tf.reduce_sum(tf.multiply(qUnderstanding, weights), axis=1)

        with tf.variable_scope("questionSummary"):
            W = tf.get_variable("W", shape=(qUnderstanding.get_shape()[-1], 1), dtype=tf.float32)
            weights = tf.map_fn(lambda uq: tf.matmul(uq, W), qUnderstanding)
            # weights = tf.reshape(weights, [-1, tf.shape(qUnderstanding)[-2]])
            # print(weights)
            weights = tf.nn.softmax(weights)
            uQ = tf.reduce_sum(tf.multiply(qUnderstanding, weights), axis=1)

        # Calculate start and end pos
        # Note: cUnderstanding shape: [batchSize, contextLen, d]
        #       qUnderstanding shape: [batchSize, ]
        #       uQ shape: [batchSize, d]
        with tf.variable_scope("startPos"):
            d = cUnderstanding.get_shape()[-1]
            W = tf.get_variable("W", shape=(d, d), dtype=tf.float32)
            # uQT = tf.reshape(uQ, [-1, tf.shape(uQ)[-1], 1])
            uQT = tf.transpose(tf.expand_dims(uQ, -1), perm=[0, 2, 1]) #shape: [batchSize, 1, d]
            uQTw = multiplyBatch(uQT, W) #shape: [batchSize, d]

            logits = tf.reduce_sum(tf.multiply(uQTw, cUnderstanding), 2) #shape: []

            # logits = tf.map_fn(lambda uc: tf.reduce_sum(tf.multiply(tf.matmul(uc, W), uQ), axis=1), cUnderstanding)
            self.startLogits, self.startProbs = maskLogits(logits, self.contextMask, 1)

        with tf.variable_scope("endPos"):
            gruInput = tf.reduce_sum(tf.multiply(tf.expand_dims(self.startProbs, axis=2), cUnderstanding), -2)
            gruInput = tf.expand_dims(gruInput, -2)
            gruSize = uQ.get_shape().as_list()[-1]
            vQ = gruWrapper(uQ, gruInput, gruSize, self.keepProb, "endPosGRU")
            vQ = tf.squeeze(vQ)

            d = cUnderstanding.get_shape()[-1]
            W = tf.get_variable("W", shape=(d, d), dtype=tf.float32)

            vQT = tf.transpose(tf.expand_dims(vQ, -1), perm=[0, 2, 1]) #shape: [batchSize, 1, d]
            vQTw = multiplyBatch(vQT, W) #shape: [batchSize, d]

            logits = tf.reduce_sum(tf.multiply(vQTw, cUnderstanding), 2) #shape: []
            # gruInput = tf.multiply(tf.expand_dims(self.startProbs, axis=2), cUnderstanding)
            # gruSize = uQ.get_shape().as_list()[-1]
            # out = gruWrapper(uQ, gruInput, gruSize, self.keepProb, "endPosGRU", mask=self.contextMask)
            # # vQT = tf.reshape(out, [-1, tf.shape(out)[-2], tf.shape(out)[-1], 1])
            # # vQT = tf.transpose(tf.expand_dims(out, -1), perm=[0, 1, 3, 2]) #shape: [batchSize, contextLen, 1, 400]
            # d = cUnderstanding.get_shape()[-1]
            # W = tf.get_variable("W", shape=(d, d), dtype=tf.float32)

            # vQTw = multiplyBatch(out, W) #shape: [batchSize, ]

            # logits = tf.reduce_sum(tf.multiply(vQTw, cUnderstanding), 2)

            self.endLogits, self.endProbs = maskLogits(logits, self.contextMask, 1)

    def addLoss(self):
        with vs.variable_scope("loss"):
                startLosses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.startLogits, labels=self.aSpans[:, 0])
                self.startLoss = tf.reduce_mean(startLosses)
                tf.summary.scalar('startLoss', self.startLoss)

                endLosses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.endLogits, labels=self.aSpans[:, 1])
                self.endLoss = tf.reduce_mean(endLosses)
                tf.summary.scalar('endLoss', self.endLoss)

                self.loss = self.startLoss + self.endLoss
                tf.summary.scalar('loss', self.loss)

    def trainStep(self, session, batch, summaryWriter):
        inputFeed = {}
        inputFeed[self.contextIds] = batch.contextIds
        inputFeed[self.contextPosIds] = batch.contextPosIds
        inputFeed[self.contextNerIds] = batch.contextNerIds
        print("train step")
        print(batch.contextNerIds.shape)
        inputFeed[self.contextMask] = batch.contextMask
        inputFeed[self.qIds] = batch.qIds
        inputFeed[self.qMask] = batch.qMask
        inputFeed[self.aSpans] = batch.aSpans
        inputFeed[self.keepProb] = 1.0 - self.FLAGS.dropout        
        
        outputFeed = [self.updates, self.summaries, self.loss, self.globalStep, self.paramNorm, self.gradientNorm]

        [_, summaries, loss, globalStep, paramNorm, gradientNorm] = session.run(outputFeed, inputFeed)

        summaryWriter.add_summary(summaries, globalStep)
        return loss, globalStep, paramNorm, gradientNorm

    def getProbs(self, session, batch):
        inputFeed = {}
        inputFeed[self.contextIds] = batch.contextIds
        inputFeed[self.contextPosIds] = batch.contextPosIds
        inputFeed[self.contextNerIds] = batch.contextNerIds
        inputFeed[self.contextMask] = batch.contextMask
        inputFeed[self.qIds] = batch.qIds
        inputFeed[self.qMask] = batch.qMask
        
        outputFeed = [self.startProbs, self.endProbs]
        [startProbs, endProbs] = session.run(outputFeed, inputFeed)
        return startProbs, endProbs
    
    def getSpans(self, session, batch):
        startProbs, endProbs = self.getProbs(session, batch)
        starts = np.argmax(startProbs, axis=1)
        ends = np.argmax(endProbs, axis=1)
        return starts, ends
      
    def getLoss(self, session, batch):
        inputFeed = {}
        inputFeed[self.contextIds] = batch.contextIds
        inputFeed[self.contextPosIds] = batch.contextPosIds
        inputFeed[self.contextNerIds] = batch.contextNerIds
        inputFeed[self.contextMask] = batch.contextMask
        inputFeed[self.qIds] = batch.qIds
        inputFeed[self.qMask] = batch.qMask
        inputFeed[self.aSpans] = batch.aSpans

        outputFeed = [self.loss]
        [loss] = session.run(outputFeed, inputFeed)

        return loss

    def checkF1(self, session, contexts, questions, answers, numSamples=100):
        numExamples = 0
        f1Sum = 0
        for batch in generateBatches(self.wordToId, contexts, questions, answers, self.FLAGS.batch_size):

            starts, ends = self.getSpans(session, batch)
            starts = starts.tolist()
            ends = ends.tolist()

            for i, (start, end, aTokens, cTokens) in enumerate(zip(starts, ends, batch.aTokens, batch.contextTokens)):
                predAnsTokens = cTokens[start : end + 1]
                predAns = " ".join(predAnsTokens)

                expAns = " ".join(aTokens)

                f1 = f1_score(predAns, expAns)
                f1Sum += f1

                if numSamples != 0 and not numExamples < numSamples:
                    break

                numExamples += 1

            if numSamples != 0 and not numExamples < numSamples:
                break

        return f1Sum / numExamples

      
    def getDevLoss(self, session, contextDev, questionDev, spansDev):
        batchLoss = []
        batchLengths = []
        
        for batch in generateBatches(self.wordToId, contextDev, questionDev, spansDev, self.FLAGS.batch_size):
            loss = self.getLoss(session, batch)  
            batchSize = batch.batchSize
            batchLoss.append(loss * batchSize)
            batchLengths.append(batchSize)
            
        numExamples = float(sum(batchLengths))
        devLoss = sum(batchLoss) / numExamples
        
        return devLoss
            

    def train(self, session, contextTrain, qTrain, spansTrain, contextDev, qDev, spansDev):
        # for batch in generateBatches(self.wordToId, contextTrain, qTrain, spansTrain, self.FLAGS.batch_size):
        #     inputFeed = {}
        #     inputFeed[self.contextIds] = batch.contextIds
        #     inputFeed[self.contextMask] = batch.contextMask
        #     inputFeed[self.qIds] = batch.qIds
        #     inputFeed[self.qMask] = batch.qMask
        #     inputFeed[self.aSpans] = batch.aSpans
        #     print(session.run([tf.shape(self.endLogits), tf.shape(self.startLogits), self.aSpans], inputFeed))
        # K.set_session(session)
        epoch = 0

        checkpointPath = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodelDir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodelCkptPath = os.path.join(bestmodelDir, "qa_best.ckpt")
        bestF1 = None

        summaryWriter = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            batchNum = 0
            print("epoch " + str(epoch))
            for batch in generateBatches(self.wordToId, contextTrain, qTrain, spansTrain, self.FLAGS.batch_size):

                loss, globalStep, paramNorm, gradientNorm = self.trainStep(session, batch, summaryWriter)
                print("step " + str(globalStep))

                if globalStep % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpointPath)
                    self.saver.save(session, checkpointPath, global_step=globalStep)
                
                # if globalStep % self.FLAGS.eval_every == 0:
                    # devLoss = self.getDevLoss(session, contextDev, qDev, spansDev)
                    
                    # print("adding summary")
                    # summary = tf.Summary()
                    # summary.value.add(tag="dev/loss", simple_value=devLoss)
                    # summaryWriter.add_summary(summary, globalStep)
                    
                    # trainF1 = self.checkF1(session, contextTrain, qTrain, spansTrain)
                    # summary = tf.Summary()
                    # summary.value.add(tag="train/F1", simple_value=trainF1)
                    # summaryWriter.add_summary(summary, globalStep)
                    
                    # print("calculating devf1")
                    # devF1 = self.checkF1(session, contextDev, qDev, spansDev, numSamples=0)
                    # summary = tf.Summary()
                    # summary.value.add(tag="dev/F1", simple_value=devF1)
                    # summaryWriter.add_summary(summary, globalStep)

                    # if bestF1 is None or devF1 > bestF1:
                    #     bestF1 = devF1
                    #     self.bestmodel_saver.save(session, bestmodelCkptPath, global_step=globalStep)
