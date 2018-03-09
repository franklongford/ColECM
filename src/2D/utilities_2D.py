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


def create_image(traj, sigma, resolution):


	histogram, xedges, yedges = np.histogram2d(traj.T[0], traj.T[1], bins=resolution)#, range=[[0, 0], [cell_dim[0], cell_dim[1]]])
	H = histogram.T

	dx = np.tile(np.arange(resolution), (resolution, 1))
	dy = np.tile(np.arange(resolution), (resolution, 1)).T

	dx -= resolution * np.array(2 * dx / resolution, dtype=int)
	dy -= resolution * np.array(2 * dy / resolution, dtype=int)

	r2 = dx**2 + dy**2
	r = np.sqrt(r2)

	r_cut = np.zeros((resolution, resolution))
	cutoff = np.where(r <= sigma * 4)
	r_cut[cutoff] += r[cutoff]

	non_zero = np.zeros((resolution, resolution))
	non_zero[cutoff] += 1
	non_zero[0][0] += 1

	indices = np.argwhere(H)
	intensity = H[np.where(H)]
	image = np.zeros((resolution, resolution))

	for i, index in enumerate(indices):

		r_cut_shift = move_2D_array_centre(r_cut, index)
		non_zero_shift = move_2D_array_centre(non_zero, index)
		image[np.where(non_zero_shift)] += gaussian(r_cut_shift[np.where(non_zero_shift)].flatten(), 0, sigma) * intensity[i]

		#r_shift = move_2D_array_centre(r, index)
		#gauss_map = np.reshape(gaussian(r_shift.flatten(), 0, sigma), 
		#				(resolution, resolution)) * intensity[i]
		#image += gauss_map

	return image
