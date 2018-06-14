"""
ColECM: Collagen ExtraCellular Matrix Simulation
MACHINE LEARNING ROUTINE 

Created by: Frank Longford
Created on: 13/06/2018

Last Modified: 13/06/2018
"""

import numpy as np
import scipy as sp

from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential, load_model # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, to_categorical # utilities for one-hot encoding of ground truth values

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os, re

import utilities as ut
import setup


def create_cnn_model(n_classes, input_shape):

	kernel_size = (3, 3) # we will use 3x3 kernels throughout
	pool_size = (2, 2) # we will use 2x2 pooling throughout
	conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
	conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
	drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
	drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
	hidden_size = 512 # the FC layer will have 512 neurons

	print(input_shape)

	model = Sequential()

	model.add(Conv2D(conv_depth_1, kernel_size, padding='same', activation='relu', input_shape=input_shape))
	#model.add(Conv2D(conv_depth_1, kernel_size, activation='relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(drop_prob_1))

	model.add(Conv2D(conv_depth_2, kernel_size, padding='same', activation='relu'))
	#model.add(Conv2D(conv_depth_2, kernel_size, activation='relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(drop_prob_1))

	model.add(Conv2D(conv_depth_2, kernel_size, padding='same', activation='relu'))
	#model.add(Conv2D(conv_depth_2, kernel_size, activation='relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(drop_prob_1))

	model.add(Flatten())
	model.add(Dense(hidden_size, activation='relu'))
	model.add(Dropout(drop_prob_2))
	model.add(Dense(n_classes, activation='softmax'))

	model.summary()	

	return model


def print_nmf_results(fig_dir, fig_name, n, title, images, n_col, n_row, image_shape):

	print('\n Creating NMF Gallery {}/{}_nmf.png'.format(fig_dir, fig_name))

	plt.figure(n, figsize=(2. * n_col, 2.26 * n_row))
	plt.suptitle(title, size=16)

	for i, comp in enumerate(images):
		plt.subplot(n_row, n_col, i + 1)
		vmax = max(comp.max(), -comp.min())
		plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
			   interpolation='nearest',
			   vmin=-vmax, vmax=vmax)
		plt.xticks(())
		plt.yticks(())

	plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
	plt.savefig('{}/{}_nmf.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def convolutional_neural_network_analysis(model_name, model_dir, predict_set, data_set=None, data_labels=None, ow_model=False):
	"""
	fourier_transform_analysis(image_shg, area, n_sample)

	Calculates fourier amplitude spectrum of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	angles:  array_like (float); shape=(n_bins)
		Angles corresponding to fourier amplitudes

	fourier_spec:  array_like (float); shape=(n_bins)
		Average Fouier amplitudes of FT of image_shg

	"""
	

	area = predict_set.shape[1]
	batch_size = 32 # in each iteration, we consider 32 training examples at once
	num_epochs = 5 # we iterate 200 times over the entire training set

	if data_set is not None:
	
		if not os.path.exists(model_dir + model_name) or ow_model:

			model = create_cnn_model(len(np.unique(data_labels)), (area, area, 1))
			model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

			model.save(model_dir + model_name)

		else: model = load_model(model_dir + model_name)
	
		n_sample = data_set.shape[0]
		data_set = data_set.reshape(data_set.shape + (1,))
		data_set /= np.max(data_set)
		data_labels = to_categorical(data_labels)

		(training_set, test_set, training_labels, test_labels) = train_test_split(data_set,
			data_labels, test_size=0.2, random_state=42)

		history = model.fit(training_set, training_labels, batch_size=batch_size, epochs=num_epochs, verbose=1)
		score = model.evaluate(test_set, test_labels)

		print(' Test Loss:', score[0])
		print(" Test Accuracy: {:.2f}%".format(score[1]*100))

		model.save(model_dir + model_name)

	else: model = load_model(model_dir + model_name)

	predict_set = predict_set.reshape(predict_set.shape + (1,))
	score = model.predict(predict_set)

	plot_classes = np.arange(score.shape[-1])
	plot_score = 100 * np.sum(score, axis=0) / np.sum(score)

	print("\n Prediction Results:")
	for i, score in enumerate(plot_score): print(" Class {}: {:.2f}%".format(plot_classes[i], score))

	return 


def nmf_analysis(data_set, n_components):
	"""
	nmf_analysis(data_set, n_sample)

	Calculates non-negative matrix factorisation of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample


	Returns
	-------

	"""

	print("\n Performing NMF Analysis")

	n_sample = data_set.shape[0]
	n_frame = data_set.shape[1]
	area = data_set.shape[2]
	rng = np.random.RandomState(0)

	model = NMF(n_components=n_components, init='random', random_state=0)

	data_set = data_set.reshape(data_set.shape[0], area**2)

	model.fit(data_set)

	nmf_components = model.components_

	return nmf_components


def learning(current_dir):

	sim_dir = current_dir + '/sim/'
	gif_dir = current_dir + '/gif/'
	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'
	model_dir = current_dir + '/model/'

	ow_shg = ('-ow_shg' in sys.argv)
	ow_mod = ('-ow_mod' in sys.argv)
	mk_gif = ('-mk_gif' in sys.argv)
	mk_nmf = ('-mk_nmf' in sys.argv)

	print("\n " + " " * 5 + "----Beginning Machine Learning Analysis----\n")
	print("\n Algorithms used:\n Non-Negative Matrix Factorisation from scikit_learn library\n Convoluted Neural Network from keras library\n")
	if not os.path.exists(gif_dir): os.mkdir(gif_dir)
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)
	if not os.path.exists(model_dir): os.mkdir(model_dir)

	train_file_names = []
	train = []
	train_ref = []

	predict_file_names = []
	predict = []

	['train', 'predict']
	n_components = 9

	if ('-name' in sys.argv): model_name = sys.argv[sys.argv.index('-name') + 1]
	else: model_name = 'colecm_cnn_model'

	if ('-train' in sys.argv):
		for arg in sys.argv[sys.argv.index('-train')+1:]: 
			if not re.match('-', arg): train_file_names.append(arg)
			else: break
	if ('-predict' in sys.argv): 
		for arg in sys.argv[sys.argv.index('-predict')+1:]: 
			if not re.match('-', arg): predict_file_names.append(arg)
			else: break

	for i, file_name in enumerate(train_file_names): 

		data_set = ut.load_npy(data_dir + file_name)
		train.append(data_set)	
		train_ref.append(np.ones(data_set.shape[0]) * i)
		
		if mk_nmf:
			"Perform Non-Negative Matrix Factorisation"
			nmf_components = nmf_analysis(data_set, n_components)
			print_nmf_results(fig_dir, file_name, 12, 'NMF Main Components', nmf_components[:n_components], 
					np.sqrt(n_components), np.sqrt(n_components), (data_set.shape[1], data_set.shape[2]))


	for file_name in predict_file_names: 

		data_set = ut.load_npy(data_dir + file_name)
		predict.append(data_set)	

		if mk_nmf:
			"Perform Non-Negative Matrix Factorisation"
			nmf_components = nmf_analysis(data_set, n_components)
			print_nmf_results(fig_dir, file_name, 12, 'NMF Main Components', nmf_components[:n_components], 
					np.sqrt(n_components), np.sqrt(n_components), (data_set.shape[1], data_set.shape[2]))


	try:
		train_data_set = np.concatenate((train))
		train_data_labels = np.concatenate((train_ref))
	except: 
		train_data_set = None
		train_data_labels = None

	predict_data_set = np.concatenate((predict))

	"Perform convolutional neural network analysis"
	convolutional_neural_network_analysis(model_name, model_dir, predict_data_set, data_set=train_data_set, data_labels=train_data_labels, ow_model=ow_mod)
