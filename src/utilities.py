"""
ColECM: Collagen ExtraCellular Matrix Simulation
UTILITIES ROUTINE 

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 12/04/2018
"""

import numpy as np
import scipy as sp

import sys, os, pickle

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)
SQRTPI = np.sqrt(np.pi)

def logo():

	print(' ' + '_' * 53)
	print( "|_______|" + ' ' * 14 + "|_______|" + ' ' * 14 + "|_______|")
	print( " \\_____/" + ' ' * 16 + "\\_____/"  + ' ' * 16 + "\\_____/")
	print( "  | | |    ___              ___   ___           | | |")
	print( "  | | |   /      ___   |   |     /    |\    /|  | | |")
	print( "  | | |  |      /   \  |   |___ |     | \  / |  | | |")
	print( "  | | |  |     |     | |   |    |     |  \/  |  | | |")
	print( "  | | |   \___  \___/  |__ |___  \___ |      |  | | |")
	print( "  |_|_|                  _____                  |_|_|")
	print( " /_____\\" + ' ' * 16 + "/_____\\"  + ' ' * 16 + "/_____\\")
	print( "|_______|" + '_' * 14 + "|_______|" + '_' * 14 + "|_______|" + '  v1.0.0.dev1')


def check_string(string, pos, sep, word):

	if sep in string: 
		temp_string = string.split(sep)
		if temp_string[pos] == word: temp_string.pop(pos)
		string = sep.join(temp_string)

	return string


def check_file_name(file_name, file_type="", extension=""):
	"""
	check_file_name(file_name, file_type="", extension="")
	
	Checks file_name for file_type or extension

	"""

	file_name = check_string(file_name, -1, '.', extension)
	file_name = check_string(file_name, -1, '_', file_type)
	
	return file_name


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



def gaussian(x, mean, std):
	"""
	Return value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)


def dx_gaussian(x, mean, std):
	"""
	Return derivative of value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return (mean - x) / std**2 * gaussian(x, mean, std)


def create_index(array):
	"""
	create_index(array)

	Takes a list of ndim indicies and returns an index array that can be used to access elements in a ndim numpy array
	"""

	ndim = array.shape[1]
	indices = ()

	for i in range(ndim): indices += (np.array(array.T[i]),)

	return indices


def check_cutoff(array, thresh):
	"""
	check_cutoff(array, rc)

	Determines whether elements of array are less than or equal to thresh
	"""

	return (array <= thresh).astype(float)


def get_distances(pos, cell_dim):
	"""
	get_distances(pos, cell_dim)

	Calculate distance vector between two beads

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions
		
	Returns
	-------

	dx:  array_like (float); shape=(n_bead, n_bead)
		Displacement along x axis between each bead

	dy:  array_like (float); shape=(n_bead, n_bead)
		Displacement along y axis between each bead

	"""

	n_bead = pos.shape[0]
	n_dim = cell_dim.shape[0]

	temp_pos = np.moveaxis(pos, 0, 1)

	dxyz = np.array([np.tile(temp_pos[0], (n_bead, 1)), np.tile(temp_pos[1], (n_bead, 1))])
	dxyz = np.reshape(np.tile(temp_pos, (1, n_bead)), (n_dim, n_bead, n_bead))
	dxyz = np.transpose(dxyz, axes=(0, 2, 1)) - dxyz

	for i in range(n_dim): dxyz[i] -= cell_dim[i] * np.array(2 * dxyz[i] / cell_dim[i], dtype=int)

	return dxyz
	

def pot_harmonic(x, x0, k): 
	"""
	pot_harmonic(x, x0, k)

	Returns harmonic potential from displacememt of x away from x0
	"""

	return k * (x - x0)**2


def force_harmonic(x, x0, k): 
	"""
	force_harmonic(x, x0, k)

	Returns force acting from displacememt of x away from x0 from harmonic potential
	"""

	return 2 * k * (x0 - x)


def pot_vdw(r2, sigma, epsilon):
	"""
	pot_vdw(x, x0, k)

	Returns Van de Waals potential from square radial distance r2
	"""
	
	return 4 * epsilon * ((sigma**2 / r2)**6 - (sigma**6 / r2**3))


def force_vdw(r2, sigma, epsilon):
	"""
	pot_harmonic(x, x0, k)

	Returns harmonic potential from displacememt of x away from x0
	"""

	return - 24 * epsilon * (2 * (sigma**2 / r2)**6 - (sigma**6 / r2**3))


def kin_energy(vel, mass, n_dim):
	"""
	kin_energy(vel)

	Returns kinetic energy of simulation in reduced units
	"""

	n_f = n_dim * (vel.shape[0] - 1) 

	return 0.5 * np.sum(mass * vel**2) / n_f


def update_bond_lists(bond_matrix):
	"""
	update_bond_lists(bond_matrix)

	Return atom indicies of angular terms
	"""

	N = bond_matrix.shape[0]

	bond_index_half = np.argwhere(np.triu(bond_matrix))
	bond_index_full = np.argwhere(bond_matrix)

	indices_half = create_index(bond_index_half)
	indices_full = create_index(bond_index_full)

	bond_beads = []
	dxdy_index = []

	count = np.unique(bond_index_full.T[0]).shape[0]

	for n in range(N):
		slice_full = np.argwhere(bond_index_full.T[0] == n)
		slice_half = np.argwhere(bond_index_half.T[0] == n)

		if slice_full.shape[0] > 1:
			bond_beads.append(np.unique(bond_index_full[slice_full].flatten()))
			dxdy_index.append(bond_index_full[slice_full][::-1])

	bond_beads = np.array(bond_beads)
	dxdy_index = np.reshape(dxdy_index, (2 * len(dxdy_index), 2))
	r_index = np.array([np.argwhere(np.sum(bond_index_half**2, axis=1) == x).flatten() for x in np.sum(dxdy_index**2, axis=1)]).flatten()

	return bond_beads, dxdy_index, r_index
