import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from graphComponents import *

class Model():

    def __init__(self, FLAGS, wordToId, idToWord, embMat):
        self.FLAGS = FLAGS
        self.wordToId = wordToId
        self.idToWord = idToWord

        with tf.variable_scope("model", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.initPlaceholders()
            self.addEmbedding(embMat)
            self.addGraph()
            self.addLoss()


    def initPlaceholders(self):
        #Context, question, and answer placeholders
        # self.contextLen = tf.placeholder(tf.int32, shape=())
        # self.qLen = tf.placeholder(tf.int32, shape=())
        self.contextIds = tf.placeholder(tf.int32, shape=[None, None])
        self.contextMask = tf.placeholder(tf.int32, shape=[None, None])
        self.qIds = tf.placeholder(tf.int32, shape=[None, None])
        self.qMask = tf.placeholder(tf.int32, shape=[None, None])
        self.aSpans = tf.placeholder(tf.int32, shape=[None, 2])

        #Dropout placeholder
        self.keepProb = tf.placeholder_with_default(1.0, shape=())

    def addEmbedding(self, embMat):
        with vs.variable_scope("embedding"):
            embeddingMatrix = tf.constant(embMat, dtype=tf.float32, name="embMat")
            self.contextEmbs = embedding_ops.embedding_lookup(embeddingMatrix, self.contextIds)
            self.qEmbs = embedding_ops.embedding_lookup(embeddingMatrix, self.qIds)

    def addGraph(self):
        # High and low level representation
        lowLevelC = biLSTM(self.contextEmbs, self.FLAGS.hidden_size, self.keepProb, "lowLevelC",mask=self.contextMask)
        # TODO: modify so that hidden size can be different between levels
        highLevelC = biLSTM(lowLevelC, self.FLAGS.hidden_size, self.keepProb, "highLevelC", mask=self.contextMask)
        lowLevelQ = biLSTM(self.qEmbs, self.FLAGS.hidden_size, self.keepProb, "lowLevelQ", mask=self.qMask)
        highLevelQ = biLSTM(lowLevelQ, self.FLAGS.hidden_size, self.keepProb, "highLevelQ", mask=self.qMask)

        # Question Understanding
        qUnderstanding = biLSTM(tf.concat([lowLevelQ, highLevelQ], axis=-1), self.FLAGS.hidden_size, self.keepProb, "qUnderstanding", mask=self.qMask)

        # Create history of word
        contextHOW = tf.concat([self.contextEmbs, lowLevelC, highLevelC], axis=-1)
        qHOW = tf.concat([self.qEmbs, lowLevelQ, highLevelQ], axis=-1)

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
        with tf.variable_scope("startPos"):
            d = cUnderstanding.get_shape()[-1]
            W = tf.get_variable("W", shape=(d, d), dtype=tf.float32)
            logits = tf.map_fn(lambda uc: tf.reduce_sum(tf.multiply(tf.matmul(uc, W), uQ), axis=1), cUnderstanding)
            self.startLogits, self.startProbs = maskLogits(logits, self.contextMask, 1)

        with tf.variable_scope("endPos"):
            gruInput = tf.multiply(tf.expand_dims(self.startProbs, axis=2), cUnderstanding)
            gruSize = uQ.get_shape().as_list()[-1]
            out = gruWrapper(uQ, gruInput, gruSize, self.keepProb, "endPosGRU", mask=self.contextMask)
            vQ = tf.reshape(out, [-1, -1, tf.shape(out)[-1], 1])

            d = cUnderstanding.get_shape()[-1]
            W = tf.get_variable("W", shape=(d, d), dtype=tf.float32)

            vQTw = multiplyBatch(vQ, W)
            print(vQ)

            logits = tf.reduce_sum(tf.multiply(vQTw, cUnderstanding), 2)

            self.endLogits, self.endProbs = maskLogits(logits, self.contextMask, 1)

        print(self.endLogits)
        print(self.startLogits)
        print(self.endProbs)
        print(self.startProbs)

    def addLoss(self):
        pass
        # with vs.variable_scope("loss"):
                # startLosses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.startLogits, labels=self.spans[:, 0])
                # self.startLoss = tf.reduce_mean(startLosses)
                # tf.summary.scalar('startLoss', self.startLoss)

                # endLosses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.endLogits, labels=self.spans[:, 1])
                # self.endLoss = tf.reduce_mean(endLosses)
                # tf.summary.scalar('endLoss', self.endLoss)

                # self.loss = self.startLoss + self.endLoss
                # tf.summary.scalar('loss', self.loss)


    def train(self, session, contextTrain, qTrain, spansTrain, contextVal, qVal, spansVal):
        pass
        # epoch = 0
        # while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
        #   epoch += 1

        #   for batch in generateBatches(self.wordToId, contextTrain, qTrain, spansTrain, self.FLAGS.batch_size):
