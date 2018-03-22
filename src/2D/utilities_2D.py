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
	"""
	move_2D_array_centre(array, centre)

	Move top left corner of 2D array to centre index
	"""

	for i, ax in enumerate([1, 0]): array = np.roll(array, centre[i], axis=ax)

	return array


def gaussian(x, mean, std):
	"""
	Return value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)


def dx_gaussian(x, mean, std):
	"""
	Return derivative of value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return (x - mean) / std**2 * gaussian(x, mean, std)


def shg_images(traj, sigma, n_x, n_y, cut):
	"""
	shg_images(traj, sigma, n_x, n_y)

	Form a set of imitation SHG images from Gaussian convolution of a set of trajectories

	Parameters
	----------

	traj:  array_like (float); shape=(n_images, n_bead, n_dim)
		Array of sampled configurations from a simulation

	sigma:  float
		Parameter to determine variance of Guassian distribution

	n_x:  int
		Number of pixels in image x dimension

	n_y:  int
		Number of pixels in image y dimension

	cut:  float
		Cutoff radius for convolution

	Returns
	-------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration
	"""

	n_image = traj.shape[0]
	image_shg = np.zeros((n_image, n_y, n_x))
	dx_shg = np.zeros((n_image, n_y, n_x))
	dy_shg = np.zeros((n_image, n_y, n_x))

	"Calculate distances between grid points"
	dx = np.tile(np.arange(n_x), (n_y, 1)).T
	dy = np.tile(np.arange(n_y), (n_x, 1))

	"Enforce periodic boundaries"
	dx -= n_x * np.array(2 * dx / n_x, dtype=int)
	dy -= n_y * np.array(2 * dy / n_y, dtype=int)

	"Calculate radial distances"
	r2 = dx**2 + dy**2

	"Find indicies within cutoff radius"
	cutoff = np.where(r2 <= cut**2)

	"Form a filter for cutoff radius"
	filter_ = np.zeros((n_x, n_y))
	filter_[cutoff] += 1

	"Get all non-zero radii"
	non_zero = np.zeros((n_x, n_y))
	non_zero[cutoff] += 1
	non_zero[0][0] = 0

	"Form a matrix of radial distances corresponding to filter" 
	r_cut = np.zeros((n_x, n_y))
	r_cut[cutoff] += np.sqrt(r2[cutoff])

	for i, image in enumerate(range(n_image)):
		hist, image_shg[i] = create_image(traj[image][0], traj[image][1], sigma, n_x, n_y, r_cut, filter_)
		dx_shg[i], dy_shg[i] = fibre_align(hist, sigma, n_x, n_y, dx, dy, r_cut, non_zero)

	return image_shg, dx_shg, dy_shg


def create_image(pos_x, pos_y, std, n_x, n_y, r_cut, filter_):
	"""
	create_image(pos_x, pos_y, sigma, n_x, n_y, r_cut, non_zero)

	Create Gaussian convoluted image from a set of bead positions

	Parameter
	---------

	pos_x:  array_like (float), shape=(n_bead)
		Bead position along x dimension

	pos_y:  array_like (float), shape=(n_bead)
		Bead position along y dimension

	std:  float
		Standard deviation of Gaussian distribution

	n_x:  int
		Number of pixels in image x dimension

	n_y:  int
		Number of pixels in image y dimension

	r_cut:  array_like (float); shape=(n_x, n_y)
		Matrix of radial distances between pixels with cutoff radius applied

	filter_:  array_like (float); shape=(n_x, n_y)
		Filter representing indicies to use in convolution

	Returns
	-------

	histogram:  array_like (int); shape=(n_x, n_y)
		Discretised distribution of pos_x and pos_y

	image:  array_like (float); shape=(n_x, n_y)
		Convoluted SHG imitation image

	"""

	"Discretise data"
	histogram, xedges, yedges = np.histogram2d(pos_x, pos_y, bins=[n_x, n_y])
	histogram = histogram.T

	"Get indicies and intensity of non-zero histogram grid points"
	indices = np.argwhere(histogram)
	intensity = histogram[np.where(histogram)]

	"Generate blank image"
	image = np.zeros((n_x, n_y))

	for i, index in enumerate(indices):
	
		r_cut_shift = move_2D_array_centre(r_cut, index)
		filter_shift = move_2D_array_centre(filter_, index)

		image[np.where(filter_shift)] += gaussian(r_cut_shift[np.where(filter_shift)].flatten(), 0, std) * intensity[i]

		#"Performs the full mapping for comparison"
		#r_shift = move_2D_array_centre(r, index)
		#gauss_map = np.reshape(gaussian(r_shift.flatten(), 0, sigma), 
		#				(n_x, n_y)) * intensity[i]
		#image += gauss_map

	image= image.T

	return histogram, image


def fibre_align(histogram, std, n_x, n_y, dx, dy, r_cut, non_zero):
	"""
	create_image(pos_x, pos_y, sigma, n_x, n_y, r_cut, non_zero)

	Create Gaussian convoluted image from a set of bead positions

	Parameter
	---------

	histogram:  array_like (int); shape=(n_x, n_y)
		Discretised distribution of pos_x and pos_y

	std:  float
		Standard deviation of Gaussian distribution

	n_x:  int
		Number of pixels in image x dimension

	n_y:  int
		Number of pixels in image y dimension

	dx:  array_like (float); shape=(n_x, n_y)
		Matrix of distances along x axis in pixels with cutoff radius applied

	dy:  array_like (float); shape=(n_x, n_y)
		Matrix of distances along y axis in pixels with cutoff radius applied

	r_cut:  array_like (float); shape=(n_x, n_y)
		Matrix of radial distances between pixels with cutoff radius applied

	non_zero:  array_like (float); shape=(n_x, n_y)
		Filter representing indicies to use in convolution

	Returns
	-------

	

	"""

	"Get indicies and intensity of non-zero histogram grid points"
	indices = np.argwhere(histogram)
	intensity = histogram[np.where(histogram)]

	"Generate blank image"
	dx_grid = np.zeros((n_x, n_y))
	dy_grid = np.zeros((n_x, n_y))

	for i, index in enumerate(indices):
	
		r_cut_shift = move_2D_array_centre(r_cut, index)
		dx_shift = move_2D_array_centre(dx, index)
		dy_shift = move_2D_array_centre(dy, index)
		non_zero_shift = move_2D_array_centre(non_zero, index)

		dx_grid[np.where(non_zero_shift)] -= (dx_gaussian(r_cut_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dx_shift[np.where(non_zero_shift)].flatten() / r_cut_shift[np.where(non_zero_shift)].flatten())
		dy_grid[np.where(non_zero_shift)] -= (dx_gaussian(r_cut_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dy_shift[np.where(non_zero_shift)].flatten() / r_cut_shift[np.where(non_zero_shift)].flatten())

	dx_grid = dx_grid.T
	dy_grid = dy_grid.T

	return dx_grid, dy_grid
