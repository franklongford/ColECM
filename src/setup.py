"""
ColECM: Collagen ExtraCellular Matrix Simulation
SETUP ROUTINE 

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 19/04/2018
"""

import numpy as np
import scipy as sp
import random

import sys, os, pickle

import utilities as ut


def get_param_defaults():
	"""
	get_param_defaults()

	Returns default simulation and analysis parameters
	"""
	
	defaults = {'n_dim' : 2,
			'n_step' : 10000,
			'save_step' : 500,
			'mass' : 1.,
			'vdw_sigma' : 1.,
			'vdw_epsilon' : 1.,
			'bond_r0' : 2.**(1./6.),
			'bond_k' : 10.,
			'angle_theta0' : np.pi,
			'angle_k' : 10.,
			'rc' : 3.0,
			'kBT' : 5.,
			'gamma' : 0.5,
			'sigma' : np.sqrt(3.75),
			'n_fibril_x' : 2,
			'n_fibril_y' : 2,
			'n_fibril_z' : 1,
			'l_fibril' : 5,
			'l_conv' : 1.,
			'res' : 7.5,
			'sharp' : 3.0,
			'skip' : 1}

	return defaults

def get_file_name_defaults():
	"""
	get_file_name_defaults()

	Returns default file names
	"""
	
	defaults = {'param_file_name' : False,
			'pos_file_name' : False,
			'traj_file_name' : False,
			'restart_file_name' : False,
			'output_file_name' : False,
			'gif_file_name' : False}

	return defaults


def check_file_names(input_list, file_names=False):
	"""
	check_file_names(input_list, file_names=False)

	Checks input_list to find file names
	"""

	if not file_names: file_names = get_file_name_defaults()

	if ('-param' in input_list): file_names['param_file_name'] = input_list[input_list.index('-param') + 1]
	if ('-pos' in input_list): file_names['pos_file_name'] =  input_list[input_list.index('-pos') + 1]

	if ('-traj' in input_list): file_names['traj_file_name'] =  input_list[input_list.index('-traj') + 1]
	else: file_names['traj_file_name'] = file_names['pos_file_name']
	if ('-rst' in input_list): file_names['restart_file_name'] = input_list[input_list.index('-rst') + 1]
	else: file_names['restart_file_name'] = file_names['pos_file_name']
	if ('-out' in input_list): file_names['output_file_name'] = input_list[input_list.index('-out') + 1]
	else: file_names['output_file_name'] = file_names['traj_file_name']
	if ('-gif' in input_list): file_names['gif_file_name'] = input_list[input_list.index('-gif') + 1]
	else: file_names['gif_file_name'] = file_names['traj_file_name']

	if file_names['param_file_name']: file_names['param_file_name'] = ut.check_file_name(file_names['param_file_name'], 'param', 'pkl') + '_param'
	if file_names['pos_file_name']: file_names['pos_file_name'] = ut.check_file_name(file_names['pos_file_name'], extension='npy')
	if file_names['traj_file_name']: file_names['traj_file_name'] = ut.check_file_name(file_names['traj_file_name'], 'traj', 'npy') + '_traj'
	if file_names['restart_file_name']: file_names['restart_file_name'] = ut.check_file_name(file_names['restart_file_name'], 'rst', 'npy') + '_rst'
	if file_names['output_file_name']: file_names['output_file_name'] = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_out'
	if file_names['gif_file_name']: file_names['gif_file_name'] = ut.check_file_name(file_names['gif_file_name'], extension='png')

	return file_names


def check_sim_param(input_list, param=False):
	"""
	check_sim_param(input_list, param=False)

	Checks input_list to overwrite simulation parameters in param dictionary 
	"""

	if not param: param = get_param_defaults()

	if ('-ndim' in input_list): param['n_dim'] = int(input_list[input_list.index('-ndim') + 1])
	if ('-mass' in input_list): param['mass'] = float(input_list[input_list.index('-mass') + 1])
	if ('-vdw_sigma' in input_list): 
		param['vdw_sigma'] = float(input_list[input_list.index('-vdw_sigma') + 1])
		param['bond_r0'] = 2.**(1./6.) * param['vdw_sigma']
	if ('-vdw_epsilon' in input_list): param['vdw_epsilon'] = float(input_list[input_list.index('-vdw_epsilon') + 1])
	if ('-bond_k' in input_list): param['bond_k'] = float(input_list[input_list.index('-bond_k') + 1])
	if ('-angle_k' in input_list): param['angle_k'] = float(input_list[input_list.index('-angle_k') + 1])
	if ('-rc' in input_list): param['rc'] = float(input_list[input_list.index('-rc') + 1])
	else: param['rc'] = param['vdw_sigma'] * 3.0
	if ('-kBT' in input_list): 
		param['kBT'] = float(input_list[input_list.index('-kBT') + 1])
		param['sigma'] = np.sqrt(param['gamma'] * (2 - param['gamma']) * (param['kBT'] / param['mass']))
	if ('-gamma' in input_list): 
		param['gamma'] = float(input_list[input_list.index('-gamma') + 1])
		param['sigma'] = np.sqrt(param['gamma'] * (2 - param['gamma']) * (param['kBT'] / param['mass']))
	if ('-nfibx' in input_list): param['n_fibril_x'] = int(input_list[input_list.index('-nfibx') + 1])
	if ('-nfiby' in input_list): param['n_fibril_y'] = int(input_list[input_list.index('-nfiby') + 1])
	if ('-nfibz' in input_list): param['n_fibril_z'] = int(input_list[input_list.index('-nfibz') + 1])
	if ('-lfib' in input_list): param['l_fibril'] = int(input_list[input_list.index('-lfib') + 1])

	return param


def check_analysis_param(input_list, param=False):
	"""
	check_analysis_param(input_list, param=False)

	Checks input_list to overwrite analysis parameters in param dictionary 
	"""

	if not param: param = get_param_defaults()

	if ('-res' in input_list): param['res'] = float(input_list[input_list.index('-res') + 1])
	if ('-sharp' in input_list): param['sharp'] = float(input_list[input_list.index('-sharp') + 1])
	if ('-skip' in input_list): param['skip'] = int(input_list[input_list.index('-skip') + 1])

	return param


def read_input_file(input_file_name, files=False, simulation=False, analysis=False, file_names=False, param=False):
	"""
	read_input_file(input_file_name, files=False, simulation=False, analysis=False, param=False)

	Opens input_file_name and checks contents for simulation and/or analysis parameters to overwrite param dictionary
	"""

	if not file_names: file_names = get_file_name_defaults()
	if not param: param = get_param_defaults()

	with open(input_file_name, 'r') as infile:
		lines = infile.read().splitlines()
	input_list = (' '.join(lines)).split()

	if ('-nstep' in input_list): param['n_step'] = int(input_list[input_list.index('-nstep') + 1])
	if ('-save_step' in input_list): param['save_step'] = int(input_list[input_list.index('-save_step') + 1])

	if files: file_names = check_file_names(input_list, file_names)
	if simulation: param = check_sim_param(input_list, param)
	if analysis: param = check_analysis_param(input_list, param)

	return file_names, param


def manual_input_param(param=False):
	"""
	manual_input_param(param=False)

	Manual paramter input (CURRENTLY OBSOLETE)
	"""

	if not param: param = get_param_defaults()

	param['mass'] = float(input("Enter bead mass: "))
	param['vdw_sigma'] = float(input("Enter vdw sigma radius: "))
	param['vdw_epsilon'] = float(input("Enter vdw epsilon energy: "))
	param['bond_k'] = float(input("Enter bond k energy: "))
	param['angle_k'] = float(input("Enter angle k energy: "))
	param['kBT'] = float(input("Enter kBT constant: "))
	param['gamma'] = float(input("Enter Langevin gamma constant: "))
	param['l_fibril']  = int(input("Enter length of fibril (no. of beads): "))
	param['n_fibril_x']  = int(input("Enter number of fibrils in x dimension: "))
	param['n_fibril_y']  = int(input("Enter number of fibrils in y dimension: "))
	if param['n_dim'] == 3: param['n_fibril_z'] = int(input("Enter number of fibrils in z dimension: "))
	param['res'] = float(input("Enter resolution (1-10): "))
	param['sharp'] = float(input("Enter sharpness (1-10): "))
	param['skip'] = int(input("Enter number of sampled frames between each png: "))

	return param


def read_shell_input(current_dir, sim_dir, input_file_name=False):
	"""
	read_shell_input(current_dir, sim_dir)

	Reads bash shell tags to gather simulation and analysis parameters.
	Order of input methods: default_param < input_file < inline tags < param_file 

	Parameters
	----------

	current_dir:  str
		Working directory from which ColECM excecutable is called

	sim_dir:  str
		Directory in which simulation files are to be saved

	Returns
	-------

	file_names:  list (str)
		List of simulation and analysis files names

	param:  dict
		Dictionary of simulation and analysis parameters
	"""

	file_names = get_file_name_defaults()
	param = get_param_defaults()

	if input_file_name: file_names, _ = read_input_file(input_file_name, files=True, file_names=file_names)

	file_names =  check_file_names(sys.argv, file_names=file_names)

	if not file_names['param_file_name']: 
		file_names['param_file_name'] = input("Enter param_file name: ")
		check_file_names(sys.argv, file_names=file_names)
	if not file_names['pos_file_name']: 
		file_names['pos_file_name'] = input("Enter pos_file name: ")
		check_file_names(sys.argv, file_names=file_names)

	keys = ['n_dim', 'mass', 'vdw_sigma', 'vdw_epsilon', 'bond_r0', 'bond_k', 'angle_theta0', 'angle_k', 'rc', 'kBT', 
			'gamma', 'l_fibril', 'n_fibril_x', 'n_fibril_y', 'n_fibril_z']

	if os.path.exists(sim_dir + file_names['param_file_name'] + '.pkl'):
		print("Loading parameter file {}.pkl".format(sim_dir + file_names['param_file_name']))
		param_file = ut.read_param_file(sim_dir + file_names['param_file_name'])
		keys += ['bond_matrix', 'vdw_matrix', 'l_conv']
		for key in keys: param[key] = param_file[key]		

	else:
		if input_file_name: _, param = read_input_file(input_file_name, simulation=True, param=param)
		param = check_sim_param(sys.argv, param)

		print("Creating parameter file {}.pkl".format(sim_dir + file_names['param_file_name'])) 
		ut.make_param_file(sim_dir + file_names['param_file_name'])

		for key in keys: ut.update_param_file(sim_dir + file_names['param_file_name'], key, param[key])
		
	assert param['n_dim'] in [2, 3]
	assert param['rc'] >= 1.5 * param['vdw_sigma']

	if ('-nstep' in sys.argv): param['n_step'] = int(sys.argv[sys.argv.index('-nstep') + 1])
	if ('-save_step' in sys.argv): param['save_step'] = int(sys.argv[sys.argv.index('-save_step') + 1])

	if input_file_name: _, param = read_input_file(input_file_name, analysis=True, param=param)
	param = check_analysis_param(sys.argv, param)	

	return file_names, param


def import_files(sim_dir, file_names, param):
	"""
	import_files(sim_dir, file_names, param)

	Imports existing or creates new simulation files listed in file_names 

	Parameters
	----------

	sim_dir:  str
		Directory in which simulation files are to be saved

	file_names:  list (str)
		List of simulation and analysis files names

	param:  dict
		Dictionary of simulation and analysis parameters

	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Velocity of each bead in all collagen fibres

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions

	param:  dict
		Dictionary of simulation and analysis parameters
	"""

	if param['n_dim'] == 2: from sim_tools_2D import create_pos_array
	elif param['n_dim'] == 3: from sim_tools_3D import create_pos_array
	
	if os.path.exists(sim_dir + file_names['restart_file_name'] + '.npy'):
		print("Loading restart file {}.npy".format(sim_dir + file_names['restart_file_name']))
		restart = ut.load_npy(sim_dir + file_names['restart_file_name'])
		pos = restart[0]
		vel = restart[1]
		cell_dim = pos[-1]
		pos = pos[:-1]

	elif os.path.exists(sim_dir + file_names['pos_file_name'] + '.npy'):
		print("Loading position file {}.npy".format(sim_dir + file_names['pos_file_name']))
		pos = ut.load_npy(sim_dir + file_names['pos_file_name'])
		cell_dim = pos[-1]
		pos = pos[:-1]
		vel = (np.random.random(pos.shape) - 0.5) * np.sqrt(2 * param['kBT'] / param['mass'])

	else:
		file_names['pos_file_name'] = ut.check_file_name(file_names['pos_file_name'], file_type='pos') + '_pos'

		print("Creating input pos file {}{}.npy".format(sim_dir, file_names['pos_file_name']))

		fibril_param = (param['l_fibril'], param['n_fibril_x'], param['n_fibril_y'], param['n_fibril_z'])
		vdw_param = (param['vdw_sigma'], param['vdw_epsilon'])
		bond_param = (param['bond_r0'], param['bond_k'])
		angle_param = (param['angle_theta0'], param['angle_k'])

		pos, cell_dim, bond_matrix, vdw_matrix = create_pos_array(param['n_dim'], fibril_param, vdw_param, 
			bond_param, angle_param, param['rc'])
		vel = np.random.normal(loc=np.sqrt(param['kBT'] / param['mass']), size=pos.shape) 

		param['bond_matrix'] = bond_matrix
		param['vdw_matrix'] = vdw_matrix
		#param['l_conv'] = 10. / (param['l_fibril'] * 2 * param['vdw_sigma'])
		param['l_conv'] = 1 / (2 * param['vdw_sigma'])

		keys = ['bond_matrix', 'vdw_matrix', 'l_conv']
		for key in keys: ut.update_param_file(sim_dir + file_names['param_file_name'], key, param[key])

		print("Saving input pos file {}{}.npy".format(sim_dir, file_names['pos_file_name']))
		ut.save_npy(sim_dir + file_names['pos_file_name'], np.vstack((pos, cell_dim)))
		
	return pos, vel, cell_dim, param


def grow_fibre(n, bead, n_dim, n_bead, pos, bond_matrix, vdw_matrix, vdw_param, bond_param, angle_param, rc, max_energy):
	"""
	grow_fibre(n, bead, n_dim, n_bead, pos, bond_matrix, vdw_param, bond_param, angle_param, rc, max_energy)

	Grow collagen fibre consisting of beads

	Parameters
	----------

	n:  int
		Index of bead in pos array

	bead: int
		Index of bead in fibre

	n_dim:  int
		Number of dimensions in simulation

	n_bead:  int
		Number of beads in simulation

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	verlet_list: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether two beads are within rc radial distance

	vdw_param: array_like (float); shape=(2)
		Sigma and epsilon paameters for Van de Waals forces

	bond_param: array_like (float); shape=(2)
		Equilibrium length and energy paameters for bonded forces

	angle_param: array_like (float); shape=(2)
		Equilibrium angle and energy paameters for angular forces

	rc:  float
		Interaction cutoff radius for non-bonded forces

	max_energy:  float
		Maximum potential energy threshold for each system configuration
		
		
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Updated positions of n_bead beads in n_dim

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Updated matrix determining whether a bond is present between two beads

	"""

	if n_dim == 2: from sim_tools_2D import calc_energy_forces
	elif n_dim == 3: from sim_tools_3D import calc_energy_forces

	cell_dim = np.array([vdw_param[0]**2 * n_bead] * n_dim)

	if bead == 0:
		pos[n] = np.random.random((n_dim)) * vdw_param[0] * 2

	else:
		bond_beads, dxy_index, r_index = ut.update_bond_lists(bond_matrix)

		energy = max_energy + 1

		while energy > max_energy:
			new_vec = ut.rand_vector(n_dim) * vdw_param[0]
			pos[n] = pos[n-1] + new_vec
			distances = ut.get_distances(pos[:bead+1], cell_dim)
			r2 = np.sum(distances**2, axis=0)

			energy, _ = calc_energy_forces(distances, r2, bond_matrix, vdw_matrix, ut.check_cutoff(r2, rc**2), 
						vdw_param, bond_param, angle_param, rc, bond_beads, dxy_index, r_index)

	return pos


def initial_state(pos, cell_dim, bond_matrix, vdw_matrix, vdw_param, bond_param, angle_param, rc, kBT):
	"""
	initial_state(pos, cell_dim, bond_matrix, vdw_matrix, vdw_param, bond_param, angle_param, rc, kBT)
	
	Calculate inital state of simulation using starting configuration and parameters provided

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	vdw_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a non-bonded vdw interaction is present between two beads

	vdw_param: array_like, dtype=float
		Parameters of van der Waals potential (sigma, epsilon)

	bond_param: array_like, dtype=float
		Parameters of bond length potential (r0, kB)

	angle_param: array_like, dtype=float
		Parameters of angular potential (theta0, kA)

	rc:  float
		Radial cutoff distance for non-bonded interactions

	kBT: float
		Value of thermostat constant kB x T in reduced units

	
	Returns
	-------

	frc: array_like, dtype=float
		Forces acting upon each bead in all collagen fibres

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance

	bond_beads:  array_like, (int); shape=(n_angle, 3)
		Array containing indicies in pos array all 3-bead angular interactions

	dxyz_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in dx and dy arrays of all bonded interactions

	r_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in r array of all bonded interactions
	
	"""

	n_bead = pos.shape[0]
	n_dim = pos.shape[1]

	if n_dim == 2: from sim_tools_2D import calc_energy_forces
	elif n_dim == 3: from sim_tools_3D import calc_energy_forces

	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)

	verlet_list = ut.check_cutoff(r2, rc**2)

	bond_beads, dxy_index, r_index = ut.update_bond_lists(bond_matrix)
	energy, frc = calc_energy_forces(distances, r2, bond_matrix, vdw_matrix, verlet_list, 
				vdw_param, bond_param, angle_param, rc, bond_beads, dxy_index, r_index)

	return energy, frc, verlet_list, bond_beads, dxy_index, r_index
