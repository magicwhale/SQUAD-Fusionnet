import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
# from tf.contrib.rnn import LSTMCell

def biLSTM(inputs, hiddenSize, keepProb, scopeName, mask=None):
	fwCell = rnn_cell.LSTMCell(hiddenSize)
	fwCell = DropoutWrapper(fwCell, input_keep_prob=keepProb)
	bwCell = rnn_cell.LSTMCell(hiddenSize)
	bwCell = DropoutWrapper(bwCell, input_keep_prob=keepProb)

	with vs.variable_scope(scopeName):
		if mask is not None:
			mask = tf.reduce_sum(mask, reduction_indices=1)			
		(fwOut, bwOut), _ = tf.nn.bidirectional_dynamic_rnn(fwCell, bwCell, inputs, mask, dtype=tf.float32)
		out = tf.concat([fwOut, bwOut], 2)
		out = tf.nn.dropout(out, keepProb)
		return out

def gruWrapper(initial, inputs, hiddenSize, keepProb, scopeName, mask=None):
	fwCell = rnn_cell.GRUCell(hiddenSize)
	fwCell = DropoutWrapper(fwCell, input_keep_prob=keepProb)

	with vs.variable_scope(scopeName):
		if mask is not None:
			mask = tf.reduce_sum(mask, reduction_indices=1)	
		out, _ = tf.nn.dynamic_rnn(fwCell, inputs, mask, initial_state=initial, dtype=tf.float32)
		out = tf.nn.dropout(out, keepProb)
		return out

def fusion(a, b, applyTo, k, keepProb, scopeName):
	with tf.variable_scope(scopeName):
		D = tf.get_variable('D', shape=[k], dtype=tf.float32)
		D = tf.diag(D)
		d_h = a.get_shape()[-1]
		U = tf.get_variable('U', shape=[d_h, k], dtype=tf.float32)

		# TODO: Check if works
		aU = tf.nn.relu(multiplyBatch(tf.nn.dropout(a, keepProb), U))
		bU = tf.nn.relu(multiplyBatch(tf.nn.dropout(b, keepProb), U))

		s_ij = tf.matmul(aU, tf.transpose(multiplyBatch(bU, D), perm=[0,2,1]))
		attention = tf.nn.softmax(s_ij, axis=2)
		return tf.matmul(attention, applyTo)

def multiplyBatch(batches, tensor):
	batchShape = tf.shape(batches)
	tensorShape = tf.shape(tensor)
	m = batchShape[-1]
	n = batchShape[-2]
	c = tensorShape[-1]
	batches = tf.reshape(batches, [-1, m])
	res = tf.matmul(batches, tensor)
	return tf.reshape(res, [-1, n , c])

def maskLogits(logits, mask, dim):
	logitMask = (1 - tf.cast(mask, 'float')) * (-1e30)
	maskedLogits = tf.add(logits, logitMask)
	probs = tf.nn.softmax(maskedLogits, dim)
	return maskedLogits, probs 
