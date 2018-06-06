"""
ColECM: Collagen ExtraCellular Matrix Simulation
UTILITIES ROUTINE 

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 12/04/2018
"""

import numpy as np

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
	print( "|_______|" + '_' * 14 + "|_______|" + '_' * 14 + "|_______|" + '  v1.3.1')
	print( "\n              Collagen ECM Simulation\n")


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


def update_param_file(param_file_name, key, value):
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
	param_file[key] = value
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


def make_earray(file_name, arrays, atom, sizes):
	"""
	make_earray(file_name, arrays, atom, sizes)

	General purpose algorithm to create an empty earray

	Parameters
	----------

	file_name:  str
		Name of file
	arrays:  str, list
		List of references for arrays in data table
	atom:  type
		Type of data in earray
	sizes:  int, tuple
		Shape of arrays in data set
	"""


	with tables.open_file(file_name, 'w') as outfile:
		for i, array in enumerate(arrays):
			outfile.create_earray(outfile.root, array, atom, sizes[i])


def make_hdf5(file_path, shape, datatype):
	"""
	make_hdf5(directory, file_name, array, shape)

	General purpose algorithm to create an empty hdf5 file

	Parameters
	----------

	file_path:  str
		Path name of hdf5 file
	shape:  int, tuple
		Shape of dataset in hdf5 file
	datatype:  type
		Data type of dataset
	"""

	shape = (0,) + shape

	make_earray(file_path + '.hdf5', ['dataset'], datatype, [shape])


def load_hdf5(file_path, frame='all'):
	"""
	load_hdf5(file_path, frame='all')

	General purpose algorithm to load an array from a hdf5 file

	Parameters
	----------

	file_path:  str
		Path name of hdf5 file
	frame:  int (optional)
		Trajectory frame to load

	Returns
	-------

	array:  array_like (float);
		Data array to be loaded, same shape as object 'dataset' in hdf5 file
	"""

	with tables.open_file(file_path + '.hdf5', 'r') as infile:
		if frame == 'all': array = infile.root.dataset[:]
		else: array = infile.root.dataset[frame]

	return array


def save_hdf5(file_path, array, frame, mode='a'):
	"""
	save_hdf5(file_path, array, dataset, frame, mode='a')

	General purpose algorithm to save an array from a single frame a hdf5 file

	Parameters
	----------

	file_path:  str
		Path name of hdf5 file
	array:  array_like (float);
		Data array to be saved, must be same shape as object 'dataset' in hdf5 file
	frame:  int
		Trajectory frame to save
	mode:  str (optional)
		Option to append 'a' to hdf5 file or overwrite 'r+' existing data	
	"""

	if not mode: return

	shape = (1,) + array.shape

	with tables.open_file(file_path + '.hdf5', mode) as outfile:
		assert outfile.root.dataset.shape[1:] == shape[1:]
		if mode.lower() == 'a':
			write_array = np.zeros(shape)
			write_array[0] = array
			outfile.root.dataset.append(write_array)
		elif mode.lower() == 'r+':
			outfile.root.dataset[frame] = array


def shape_check_hdf5(file_path):
	"""
	shape_check_hdf5(file_path)

	General purpose algorithm to check the shape the dataset in a hdf5 file 

	Parameters
	----------

	file_path:  str
		Path name of hdf5 file

	Returns
	-------

	shape_hdf5:  int, tuple
		Shape of object dataset in hdf5 file
	"""

	with tables.open_file(file_path + '.hdf5', 'r') as infile:
		shape_hdf5 = infile.root.dataset.shape

	return shape_hdf5


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


def unit_vector(vector, axis=-1):
	"""
	unit_vector(vector, axis=-1)

	Returns unit vector of vector
	"""

	vector = np.array(vector)
	magnitude_2 = np.resize(np.sum(vector**2, axis=axis), vector.shape)
	u_vector = np.sqrt(vector**2 / magnitude_2) * np.sign(vector)

	return u_vector


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


def reorder_array(array):
	"""
	reorder_array(array)

	Inverts 3D array so that outer loop is along z axis
	"""

	return np.moveaxis(array, (2, 0, 1), (0, 1, 2))


def move_array_centre(array, centre):
	"""
	move_array_centre(array, centre)

	Move top left corner of ND array to centre index
	"""

	n_dim = centre.shape[0]

	for i, ax in enumerate(range(n_dim)): array = np.roll(array, centre[i], axis=ax)

	return array


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

	return (array <= thresh)


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

	dxyz = np.reshape(np.tile(temp_pos, (1, n_bead)), (n_dim, n_bead, n_bead))
	dxyz = np.transpose(dxyz, axes=(0, 2, 1)) - dxyz

	for i in range(n_dim): dxyz[i] -= cell_dim[i] * np.array(2 * dxyz[i] / cell_dim[i], dtype=int)

	return dxyz


def get_distances_mpi(pos, indices, cell_dim):
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
	n_bead_proc = indices.shape[0]
	n_dim = cell_dim.shape[0]

	temp_pos_1 = np.moveaxis(pos, 0, 1)
	temp_pos_2 = np.moveaxis(pos[indices], 0, 1)

	dxyz_1 = np.reshape(np.tile(temp_pos_1, (1, n_bead_proc)), (n_dim, n_bead_proc, n_bead))
	dxyz_2 = np.repeat(temp_pos_2, n_bead, axis=1).reshape((n_dim, n_bead_proc, n_bead))
	dxyz = dxyz_2 - dxyz_1

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

	return 0.5 * np.sum(mass * vel**2)


def update_bond_lists_mpi(bond_matrix, comm, size, rank):
	"""
	update_bond_lists(bond_matrix)

	Return atom indicies of angular terms
	"""

	N = bond_matrix.shape[0]

	"Get indicies of bonded beads"
	bond_index_full = np.argwhere(bond_matrix)

	"Create index lists for referring to in 2D arrays"
	indices_full = create_index(bond_index_full)

	angle_indices = []
	angle_bond_indices = []

	"Count number of unique bonds"
	count = np.unique(bond_index_full.T[0]).shape[0]

	"""
	"Find indicies of ends of fibrils"
	fib_end_check = np.argwhere(np.sum(bond_matrix, axis=1) <= 1)
	n_fib_end = fib_end_check.shape[0]
	fib_end_check_ind = np.tile(fib_end_check, n_fib_end)
	fib_end_check_ind = np.stack((fib_end_check_ind, fib_end_check_ind.T), axis=2)
	fib_end_check_ind = create_index(fib_end_check_ind[np.where(~np.eye(n_fib_end,dtype=bool))])

	fib_end = np.zeros(bond_matrix.shape)
	fib_end[fib_end_check_ind] += 1
	"""

	for n in range(N):
		slice_full = np.argwhere(bond_index_full.T[0] == n)

		if slice_full.shape[0] > 1:
			angle_indices.append(np.unique(bond_index_full[slice_full].flatten()))
			angle_bond_indices.append(bond_index_full[slice_full][::-1])

	bond_indices = np.nonzero(np.array_split(bond_matrix, size)[rank])
	angle_indices = np.array_split(angle_indices, size)[rank]
	angle_bond_indices = create_index(np.array_split(angle_bond_indices, size)[rank].reshape((2 * len(angle_indices), 2)))
	
	return bond_indices, angle_indices, angle_bond_indices


def update_bond_lists(bond_matrix):
	"""
	update_bond_lists(bond_matrix)

	Return atom indicies of angular terms
	"""

	N = bond_matrix.shape[0]

	"Get indicies of bonded beads"
	bond_index_full = np.argwhere(bond_matrix)

	"Create index lists for referring to in 2D arrays"
	indices_full = create_index(bond_index_full)

	angle_indices = []
	angle_bond_indices = []

	"Count number of unique bonds"
	count = np.unique(bond_index_full.T[0]).shape[0]

	"""
	"Find indicies of ends of fibrils"
	fib_end_check = np.argwhere(np.sum(bond_matrix, axis=1) <= 1)
	n_fib_end = fib_end_check.shape[0]
	fib_end_check_ind = np.tile(fib_end_check, n_fib_end)
	fib_end_check_ind = np.stack((fib_end_check_ind, fib_end_check_ind.T), axis=2)
	fib_end_check_ind = create_index(fib_end_check_ind[np.where(~np.eye(n_fib_end,dtype=bool))])

	fib_end = np.zeros(bond_matrix.shape)
	fib_end[fib_end_check_ind] += 1
	"""

	for n in range(N):
		slice_full = np.argwhere(bond_index_full.T[0] == n)

		if slice_full.shape[0] > 1:
			angle_indices.append(np.unique(bond_index_full[slice_full].flatten()))
			angle_bond_indices.append(bond_index_full[slice_full][::-1])

	bond_indices = np.nonzero(bond_matrix)
	angle_indices = np.array(angle_indices)
	angle_bond_indices = create_index(np.array(angle_bond_indices).reshape((2 * len(angle_indices), 2)))

	return bond_indices, angle_indices, angle_bond_indices


def centre_of_mass(pos, mass, n_fibril, l_fibril, n_dim):
	"""
	centre_of_mass(pos, mass, n_fibril, l_fibril, n_dim)

	Calculate fibrillar centre of mass

	Parameters:
	-----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	mass:  float
		Mass of fibril bead

	n_fibril:  int
		Number of fibrils in simulation

	l_fibril:  int
		Length of fibrils in simulation

	n_dim:  int
		Number of dimensions

	Returns:
	--------

	com:  array_like (float); shape=(n_dim, n_fibril)
		Array of centre of mass for each fibril
 
	"""

	com = np.zeros((n_dim, n_fibril))

	for i in range(n_dim): com[i] += np.sum(np.reshape(pos.T[i] * mass, (n_fibril, l_fibril)), axis=1) / (l_fibril * mass)

	return com


def bond_check(bond_matrix, fib_end, r2, rc, bond_rb, vdw_sigma):

	verlet_list_rb = check_cutoff(r2, bond_rb**2)

	bond_break_check = np.logical_not(verlet_list_rb) * bond_matrix
	
	bond_form_prob = fib_end * verlet_list_rb * (r2 - vdw_sigma**2) / (bond_rb**2 - vdw_sigma**2)
	bond_form_check = np.array(fib_end * np.triu(bond_form_prob) + np.triu(np.random.random(bond_matrix.shape)), dtype=int)
	bond_form_check += np.triu(bond_form_check).T
	if np.any(np.sum(abs(bond_form_check), axis=1) > 1):
		bond_form_check[np.argwhere(np.sum(bond_form_check, axis=1) > 1)] = 0
	bond_check = bond_form_check - bond_break_check

	if np.count_nonzero(bond_check) > 0:
		bond_matrix += bond_check
		return bond_matrix, True

	else: return bond_matrix, False

