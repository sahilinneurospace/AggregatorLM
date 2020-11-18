from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, InputLayer, Dense, Reshape, Flatten, Dropout, Concatenate, Average, Multiply, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.misc

import sys
from glob import glob
import os
import scipy
import random
from keras import backend as K
from keras.layers import Layer, AveragePooling2D, Reshape, Flatten
from keras.initializers import Constant

import numpy as np

def pgan_loss(y_true, y_pred):
	print(y_true, y_pred)
	age_true, race_true, y_true = y_true[:, 1], y_true[:, 2:], y_true[:, 0]
	age_pred, race_pred, y_pred = y_pred[:, 1], y_pred[:, 2:], y_pred[:, 0]
	age_loss = tf.reduce_mean(tf.to_float(tf.less(age_true, 20))*tf.maximum(tf.abs(age_pred-age_true)-5, 0) + tf.to_float(tf.logical_and(tf.greater_equal(age_true,20), tf.less(age_true,55)))*tf.maximum(tf.abs(age_pred-age_true)-10, 0) + tf.to_float(tf.logical_and(tf.greater_equal(age_true,55), tf.less(age_true,85)))*tf.maximum(tf.abs(age_pred-age_true)-20, 0) + tf.to_float(tf.greater_equal(age_true,85))*tf.maximum(tf.abs(age_pred-age_true)-30, 0))
	race_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=race_true, logits=race_pred))
	return -(1-y_true)*K.log(1-y_pred) - y_true*(K.log(y_pred)+y_pred*0.5*(age_loss+race_loss))

class FixedWeightConv2D(Layer):

	def __init__(self, **kwargs):
		super(FixedWeightConv2D, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		super(FixedWeightConv2D, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		assert isinstance(x, list)
		a, b = x
		L = []
		for i in range(a.shape[0]):
			L.append(K.squeeze(K.conv2d(K.expand_dims(a[i], axis=0), b[i], padding='same'), axis=0))
		return K.stack(L, axis=0)

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		shape_a, shape_b = input_shape
		return (shape_a[0], shape_a[1], shape_a[2], shape_b[-1])

class ParaGAN():
	
	def __init__(self, g_num_layers=3, g_init_frames=512, g_init_dim=8, d_num_layers=3, d_final_frames=512, batch_size=50):
		# Input shape
		self.g_num_layers = g_num_layers
		self.g_init_frames = g_init_frames
		self.g_init_dim = g_init_dim
		self.d_num_layers = d_num_layers
		self.d_final_frames = d_final_frames		
		self.img_rows = g_init_dim * 2**g_num_layers
		self.img_cols = g_init_dim * 2**g_num_layers
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 256
		self.num_features = 6
		self.batch_size = batch_size
		

		optimizer = Adam(0.0002, 0.5, clipnorm=0.2)
		
		# Build the generator
		self.generator = self.build_generator(g_num_layers, g_init_frames, g_init_dim)
		self.generator.summary()
		
		# Build the discriminator
		self.discriminator = self.build_discriminator(d_num_layers, d_final_frames)
		self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
		
		# Build the age estimator
		self.age_estimator = self.build_age_estimator()
		#self.age_estimator.load_weights('checkpoints/agenet-05-1.63.hdf5')
		self.age_estimator.trainable = False
		
		# Build the race estimator
		self.race_estimator = self.build_race_estimator()
		#self.race_estimator.load_weights('checkpoints/racenet-05-0.76.hdf5')
		self.race_estimator.trainable = False
		
		# The generator takes noise as input and generates imgs
		features = Input(shape=(self.num_features,), batch_shape=(self.batch_size, self.num_features))
		latent_vec = Input(shape=(self.latent_dim,), batch_shape=(self.batch_size, self.latent_dim))
		img = self.generator([features, latent_vec])
		age = self.age_estimator(img)
		race = self.race_estimator(img)
		self.discriminator.trainable = False
		valid = self.discriminator(img)
		
		self.paragan = Model([features, latent_vec], Concatenate()([valid, age, race]))
		self.paragan.summary()
		self.paragan.compile(loss=pgan_loss, optimizer=optimizer)


	def build_generator(self, g_num_layers, g_init_frames, g_init_dim):

		features = Input(shape=(self.num_features,), batch_shape=(self.batch_size, self.num_features))
		latent_vec = Input(shape=(self.latent_dim,), batch_shape=(self.batch_size, self.latent_dim))

		x = Dense(g_init_frames * g_init_dim * g_init_dim, activation="relu", input_dim=self.latent_dim)(latent_vec)
		x = Reshape((g_init_dim, g_init_dim, g_init_frames))(x)
		
		frames = g_init_frames
		
		for _ in range(g_num_layers):
			x = UpSampling2D()(x)
			conv = BatchNormalization(gamma_initializer=Constant(0.2/(1.5*frames)**0.5))(Reshape((5, 5, frames, int(frames/2)))(Dense(5*5*frames*int(frames/2))(Dense(1)(Dense(self.num_features)(features)))))
			x = FixedWeightConv2D()([x, conv])
			x = Activation('relu')(x)
			x = BatchNormalization(momentum=0.8)(x)
			frames = int(frames/2)
		
		conv = BatchNormalization(gamma_initializer=Constant(0.2))(Reshape((5, 5, frames, self.channels))(Dense(5*5*frames*self.channels)(Dense(1)(Dense(self.num_features)(features)))))
		img = FixedWeightConv2D()([x, conv])

		return Model([features, latent_vec], img)
	
	def build_discriminator(self, d_num_layers, d_final_frames):

		model = Sequential()
		
		frames = int(d_final_frames/2**d_num_layers)
		
		model.add(InputLayer(input_shape=(self.img_rows, self.img_cols, self.channels), batch_size=self.batch_size))
		model.add(Conv2D(frames, kernel_size=3, strides=2, padding="same", input_shape=(self.img_rows, self.img_cols, self.channels)))
		model.add(LeakyReLU(0.2))
		
		for _ in range(d_num_layers-1):
			model.add(Conv2D(frames, kernel_size=3, strides=2, padding="same"))
			model.add(LeakyReLU(0.2))
			model.add(BatchNormalization(momentum=0.8))
			frames = frames * 2
		
		model.add(Flatten())
		model.add(Dense(1, activation="sigmoid"))
		
		return model
		
	def build_age_estimator(self):
		
		x_age = Input(shape=self.img_shape)
		conv1_age = Conv2D(32, (3,3), padding="same")(x_age)
		maxp1_age = MaxPooling2D(strides=(2,2))(conv1_age)
		conv2_age = Conv2D(64, (3,3), padding="same")(maxp1_age)
		maxp2_age = MaxPooling2D(strides=(2,2))(conv2_age)
		conv3_age = Conv2D(128, (3,3), padding="same")(maxp2_age)
		maxp3_age = MaxPooling2D(strides=(2,2))(conv3_age)
		l1_age = Flatten()(maxp1_age)
		l2_age = Flatten()(maxp2_age)
		l3_age = Flatten()(maxp3_age)
		fc_age = Concatenate()([l1_age, l2_age, l3_age])
		age = Dense(1)(fc_age)
		
		return Model(inputs=x_age, outputs=age)
		
	def build_race_estimator(self):
		
		x_race = Input(shape=self.img_shape)
		conv1_race = Conv2D(32, (3,3), padding="same")(x_race)
		maxp1_race = MaxPooling2D(strides=(2,2))(conv1_race)
		conv2_race = Conv2D(64, (3,3), padding="same")(maxp1_race)
		maxp2_race = MaxPooling2D(strides=(2,2))(conv2_race)
		conv3_race = Conv2D(128, (3,3), padding="same")(maxp2_race)
		maxp3_race = MaxPooling2D(strides=(2,2))(conv3_race)
		l1_race = Flatten()(maxp1_race)
		l2_race = Flatten()(maxp2_race)
		l3_race = Flatten()(maxp3_race)
		fc_race = Concatenate()([l1_race, l2_race, l3_race])
		race_logits = Dense(5)(fc_race)
		
		return Model(inputs=x_race, outputs=race_logits)

	def train(self, epochs=1, n_discriminator_update=1, n_generator_update=2, batch_size=64, save_interval=500):

		data = glob(os.path.join("../faces/*"))
		
		for epoch in range(epochs):
			random.shuffle(data)
			for idx in range(int(len(data)/batch_size)):
				batch_files = data[idx*batch_size:(idx+1)*batch_size]
				batch = [np.array(scipy.misc.imresize(scipy.misc.imread(batch_file), (self.img_rows, self.img_cols))) for batch_file in batch_files]
				batch = [x/127.5 - 1 for x in batch]
				batch = np.array(batch).astype(np.float32)
				ages = np.random.uniform(0, 125, (batch_size,1))
				race_probs_ = np.random.randint(0, 1e5, (batch_size, 5))
				race_probs = np.array([race_probs_[i] / np.max(race_probs_, 1)[i] for i in range(batch_size)])
				f_in = np.concatenate((ages, race_probs), 1)
				rand_noise = np.random.uniform(0,1,(batch_size, self.latent_dim))
				# Update D network
				for _ in range(n_discriminator_update):
					imgs = self.generator.predict([f_in, rand_noise])
					errD_fake = self.discriminator.train_on_batch(batch, np.ones(batch_size))
					errD_real = self.discriminator.train_on_batch(imgs, np.zeros(batch_size))

				# Update G network
				for _ in range(n_generator_update):
					errG = self.paragan.train_on_batch([f_in, rand_noise], np.concatenate((np.ones(batch_size), ages, race_probs)))
				
				print("Epoch {:2d} [{:2d}]/[{:2d}]: errD_fake = {:.4f}, errD_real = {:.4f}, errG = {:.4f}".format(epoch, idx, int(len(data)/batch_size), errD_fake, errD_real, errG))

class Chutiya(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(Chutiya, self).__init__(**kwargs)

	def build(self, input_shape):
		assert isinstance(input_shape, list)
		super(Chutiya, self).build(input_shape)

	def call(self, x):
		assert isinstance(x, list)
		infovecs, sequence = x
		return infovecs

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		return input_shape[0]

if __name__ == '__main__':
	paragan = ParaGAN()
	infovecs = Input(shape=(128,), batch_shape=(32,128))
	sequence = Input(shape=(32,128), batch_shape=(1,32,128))
	out = Chutiya()([infovecs, sequence])
	print(out)
	m = Model([infovecs, sequence], out)