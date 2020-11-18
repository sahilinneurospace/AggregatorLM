import tensorflow as tf
from tensorflow.python.keras.layers import Input, TimeDistributed, Reshape, Dense, Lambda, Activation
from tensorflow.python.keras.models import Model
import keras.backend as K

class FixedWeightConv1D(tf.keras.layers.Layer):

	def __init__(self, first=False, **kwargs):
		self.first = first
		super(FixedWeightConv1D, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		super(FixedWeightConv1D, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		assert isinstance(x, list)
		a, b = x
		L = []
		for i in range(b.shape[0]):
			if self.first:
				L.append(K.squeeze(K.conv1d(a, b[i], padding='same', strides=2), axis=0))
			else:
				L.append(K.squeeze(K.conv1d(K.expand_dims(a[i], axis=0), b[i], padding='same', strides=2), axis=0))
		return K.stack(L, axis=0)

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		shape_a, shape_b = input_shape
		return (shape_b[0], shape_a[1], shape_b[-1])

class FixedWeightDense(tf.keras.layers.Layer):

	def __init__(self, **kwargs):
		super(FixedWeightDense, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		super(FixedWeightDense, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		assert isinstance(x, list)
		a, b = x
		L = []
		for i in range(b.shape[0]):
			L.append(K.squeeze(K.dot(K.expand_dims(a[i], axis=0), b[i]), axis=0))
		return K.stack(L, axis=0)

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		shape_a, shape_b = input_shape
		return (shape_b[0],) + shape_a[1:-1] + (shape_b[-1])


class ConvInfoGatherer():
	"""
	Takes required information vector and avaiable information sequence of vectors as inputs.
	Extracts information from the sequence of available information vectors and returns the results.
	"""

	def __init__(self,
		output_vec_size = 256,
		num_info_heads = 2,
		diff_depth_per_head = False,
		depth_per_head = 3,
		filter_size = 5,
		conv_activation = 'relu',
		output_activation = 'relu',
		seq_length = 128,
		inp_embedding_size = 128,
		choke_factor = 16):

		self.output_vec_size = output_vec_size
		self.num_info_heads = num_info_heads
		self.diff_depth_per_head = diff_depth_per_head
		self.depth_per_head = depth_per_head
		if not diff_depth_per_head:
			self.depth_per_head = [depth_per_head] * num_info_heads
		self.filter_size = filter_size
		self.conv_activation = conv_activation
		self.output_activation = output_activation
		self.seq_length = seq_length
		self.inp_embedding_size = inp_embedding_size
		self.choke_factor = choke_factor

		infovecs = Input(shape=(inp_embedding_size,), batch_size=seq_length)
		sequence_ = Input(shape=(inp_embedding_size), batch_size=seq_length)

		sequence = K.expand_dims(sequence_, axis=0)

		output = []

		for i in range(self.num_info_heads):
			y = sequence
			first = True
			seq_length = self.seq_length
			for j in range(self.depth_per_head[i]):
				choke_layer = Dense(int(self.inp_embedding_size/self.choke_factor), 
					name='dense_conv_gen_choke_'+str(i)+'_'+str(j), activation='relu')
				kernel_layer = Dense(self.filter_size*inp_embedding_size*2**j*inp_embedding_size*2**(j+1), 
					name='dense_conv_gen_'+str(i)+'_'+str(j), activation='tanh')
				conv_kernels = Reshape((self.seq_length, self.filter_size, inp_embedding_size*2**j, 
					inp_embedding_size*2**(j+1)))(TimeDistributed(kernel_layer)(TimeDistributed(choke_layer)(K.expand_dims(infovecs, axis=0))))
				conv_kernels = K.squeeze(conv_kernels, axis=0)
				y = Activation(self.conv_activation)(FixedWeightConv1D(first=first)([y, conv_kernels]))
				first = False
				seq_length = int(seq_length/2)
			choke_layer = Dense(1, 
				name='dense_dense_gen_choke_'+str(i), activation='relu')
			kernel_layer = Dense(seq_length*inp_embedding_size*2**self.depth_per_head[i]*self.output_vec_size, 
				name='dense_dense_gen_'+str(i), activation='tanh')
			dense_kernels = Reshape((self.seq_length, seq_length*inp_embedding_size*2**self.depth_per_head[i], 
				self.output_vec_size))(TimeDistributed(kernel_layer)(TimeDistributed(choke_layer)(K.expand_dims(infovecs, axis=0))))
			dense_kernels = K.squeeze(dense_kernels, axis=0)
			y = K.reshape(y, (self.seq_length, -1))
			y = Activation(self.output_activation)(FixedWeightDense()([y, dense_kernels]))
			output.append(y)

		output = Lambda(lambda x: K.stack(x, axis=1))(output)

		self.model = Model([infovecs, sequence_], output)
		self.model.summary()

	def __call__(self, x):
		return self.model(x)

class ConvInfoGathererLayer(tf.keras.layers.Layer):

	def __init__(self,
		output_vec_size = 256,
		num_info_heads = 2,
		diff_depth_per_head = False,
		depth_per_head = 3,
		filter_size = 5,
		conv_activation = 'relu',
		output_activation = 'relu',
		seq_length = 32,
		inp_embedding_size = 16,
		**kwargs):

		self.output_vec_size = output_vec_size
		self.num_info_heads = num_info_heads
		self.diff_depth_per_head = diff_depth_per_head
		self.depth_per_head = depth_per_head
		self.filter_size = filter_size
		self.conv_activation = conv_activation
		self.output_activation = output_activation
		self.seq_length = seq_length
		self.inp_embedding_size = inp_embedding_size
		super(ConvInfoGathererLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.convInfoGatherer = ConvInfoGatherer(output_vec_size = self.output_vec_size,
			num_info_heads = self.num_info_heads,
			diff_depth_per_head = self.diff_depth_per_head,
			depth_per_head = self.depth_per_head,
			filter_size = self.filter_size,
			conv_activation = self.conv_activation,
			output_activation = self.output_activation,
			seq_length = self.seq_length,
			inp_embedding_size = self.inp_embedding_size)
		super(ConvInfoGathererLayer, self).build(input_shape)

	def call(self, x):
		infovecs, sequence = x
		infovecs = tf.unstack(infovecs, axis=0)
		sequence = tf.unstack(sequence, axis=0)
		output = []
		for a, b in zip(infovecs, sequence):
			output.append(self.ConvInfoGatherer([a, b]))
		return K.stack(output, axis=0)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.num_info_heads, self.output_vec_size)


if __name__ == '__main__':
	m = ConvInfoGatherer()
	x = Input(shape=(128, 128,), batch_size=4)
	y = Input(shape=(128, 128,), batch_size=4)
	infovecs = tf.unstack(x, axis=0)
	sequence = tf.unstack(y, axis=0)
	output = []
	for a, b in zip(infovecs, sequence):
		output.append(m([a, b]))
	output = K.stack(output, axis=0)
	m = Model([x, y], output)
	m.summary()