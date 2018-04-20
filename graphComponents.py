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
