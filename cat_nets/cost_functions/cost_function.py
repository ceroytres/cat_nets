from __future__ import division

import tensorflow as tf

def perceptron_cost(y,f):

	with name_scope(scope,"perceptron_loss",[y,f]):
		return tf.reduce_mean(tf.nn.relu(-y*f))


