"""
COLLAGEN FIBRE SIMULATION 2D

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 09/03/2018
"""

import numpy as np
import scipy as sp
import random

import sys, os, pickle


SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)
SQRTPI = np.sqrt(np.pi)



def make_param_file(param_file_name):
	"""
	make_paramfile(paramfile_name)

	Creates checkfile for analysis, storing key paramtere
	"""

	param_file = {}
	pickle.dump(param_file, open(param_file_name + '.pkl', 'wb'))


def read_param_file(param_file_name):
	"""
	read_param_file(param_file_name)

	Reads param_file to lookup stored key paramters
	"""

	param_file = pickle.load(open(param_file_name + '.pkl', 'rb'))
	return param_file


def update_param_file(param_file_name, symb, obj):
	"""
	update_paramfile(param_file_name, symb, obj)

	Updates checkfile parameter

	Parameters
	----------
	paramfile_name:  str
			Paramfile path + name
	symb:  str
			Key for paramfile dictionary of object obj
	obj:
			Parameter to be saved

	Returns
	-------
	paramfile:  dict
			Dictionary of key parameters
	"""

	param_file = pickle.load(open(param_file_name + '.pkl', 'rb'))
	param_file[symb] = obj
	pickle.dump(param_file, open(param_file_name + '.pkl', 'wb'))

	return param_file


def save_npy(file_path, array):
	"""
	save_npy(file_path, array)

	General purpose algorithm to save an array to a npy file

	Parameters
	----------

	file_path:  str
		Path name of npy file
	array:  array_like (float);
		Data array to be saved
	"""

	np.save(file_path, array)


def load_npy(file_path, frames=[]):
	"""
	load_npy(file_path, frames=[])

	General purpose algorithm to load an array from a npy file

	Parameters
	----------

	file_path:  str
		Path name of npy file
	frames:  int, list (optional)
		Trajectory frames to load

	Returns
	-------
	array:  array_like (float);
		Data array to be loaded
	"""

	if len(frames) == 0: array = np.load(file_path + '.npy')
	else: array = np.load(file_path + '.npy')[frames]

	return array


def numpy_remove(list1, list2):
	"""
	numpy_remove(list1, list2)

	Deletes overlapping elements of list2 from list1
	"""

	return np.delete(list1, np.where(np.isin(list1, list2)))


def cum_mov_average(array):
	"""
	cum_mov_average(array)
	
	Returns cumulative moving average of array elements
	"""

	l = len(array)
	average = np.zeros(l)
	average[0] = array[0]

	for i in range(l-1):
		average[i+1] = average[i] + (array[i+1] - average[i]) / (i+1)  
	
	return average


def unit_vector(vector):
	"""
	unit_vector(vector)
	
	Returns unit vector of input vector

	"""
	vector_2 = vector**2 
	norm = 1. / np.sum(vector_2)

	unit_vector = np.sqrt(vector_2 * norm) * np.sign(vector) 

	return unit_vector


def rand_vector(n): 
	"""
	rand_vector(n)
	
	Returns n dimensional unit vector, components of which lie in the range -1..1

	"""

	return unit_vector(np.random.random((n)) * 2 - 1) 


def remove_element(a, array): 
	"""
	remove_element(a, array)
	
	Returns new array without element a

	"""

	return np.array([x for x in array if x != a])


def create_index(array):
	"""
	create_index(array)

	Takes a list of 2D indicies and returns an index array that can be used to access elements in a 2D numpy array
	"""
	
	return (np.array(array.T[0]), np.array(array.T[1]))


def move_2D_array_centre(array, centre):

	for i, ax in enumerate([1, 0]): array = np.roll(array, centre[i], axis=ax)

	return array


def gaussian(x, mean, std): return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)



def images_for_gif(traj, sigma, n_x, n_y, n_image):

	image_shg = np.zeros((n_image, n_y, n_x))

	"Calculate distances between grid points"
	dx = np.tile(np.arange(n_x), (n_y, 1)).T
	dy = np.tile(np.arange(n_y), (n_x, 1))

	"Enforce periodic boundaries"
	dx -= n_x * np.array(2 * dx / n_x, dtype=int)
	dy -= n_y * np.array(2 * dy / n_y, dtype=int)

	"Calculate radial distances"
	r2 = dx**2 + dy**2

	"Find indicies within cutoff radius"
	cutoff = np.where(r2 <= (sigma * 2)**2)

	"Form a filter for cutoff radius"
	non_zero = np.zeros((n_x, n_y))
	non_zero[cutoff] += 1
	non_zero[0][0] = 1

	"Form a matrix of radial distances corresponding to filter" 
	r_cut = np.zeros((n_x, n_y))
	r_cut[cutoff] += np.sqrt(r2[cutoff])

	for i, image in enumerate(range(n_image)):
		_, image_shg[i] = create_image(traj[image][0], traj[image][1], sigma, n_x, n_y, r_cut, non_zero)

	return image_shg


def create_image(pos_x, pos_y, sigma, n_x, n_y, r_cut, non_zero):

	"Discretise data"
	histogram, xedges, yedges = np.histogram2d(pos_x, pos_y, bins=[n_x, n_y])#, range=[[0, 0], [cell_dim[0], cell_dim[1]]])
	H = histogram.T

	"Get indicies and intensity of non-zero histogram grid points"
	indices = np.argwhere(H)
	intensity = H[np.where(H)]

	"Generate blank image"
	image = np.zeros((n_x, n_y))

	for i, index in enumerate(indices):
	
		r_cut_shift = move_2D_array_centre(r_cut, index)
		non_zero_shift = move_2D_array_centre(non_zero, index)
		image[np.where(non_zero_shift)] += gaussian(r_cut_shift[np.where(non_zero_shift)].flatten(), 0, sigma) * intensity[i]

		#"Performs the full mapping for comparison"
		#r_shift = move_2D_array_centre(r, index)
		#gauss_map = np.reshape(gaussian(r_shift.flatten(), 0, sigma), 
		#				(n_x, n_y)) * intensity[i]
		#image += gauss_map


	return histogram.T, image.T

