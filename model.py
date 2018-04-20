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
		highLevelC = biLSTM(lowLevelC, self.FLAGS.hidden_size, self.keepProb, "highLevelC")
		lowLevelQ = biLSTM(self.qEmbs, self.FLAGS.hidden_size, self.keepProb, "lowLevelQ", mask=self.qMask)
		highLevelQ = biLSTM(lowLevelQ, self.FLAGS.hidden_size, self.keepProb, "highLevelQ")

		# Question Understanding
		qUnderstanding = biLSTM(tf.concat([lowLevelQ, highLevelQ], axis=-1), self.FLAGS.hidden_size, self.keepProb, "qUnderstanding")

		# Create history of word
		contextHOW = tf.concat([self.contextEmbs, lowLevelC, highLevelC], axis=-1)
		qHOW = tf.concat([self.qEmbs, lowLevelQ, highLevelQ], axis=-1)

		

	# def trainStep(self, session, batch, summaryWriter):
		

	def train(self, session, contextTrain, qTrain, spansTrain, contextVal, qVal, spansVal):

		epoch = 0
		while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
			epoch += 1

			for batch in generateBatches(self.wordToId, contextTrain, qTrain, spansTrain, self.FLAGS.batch_size):
