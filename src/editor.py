"""
ColECM: Collagen ExtraCellular Matrix Simulation
FILE EDITOR ROUTINE 

Created by: Frank Longford
Created on: 15/05/2018

Last Modified: 15/05/2018
"""

import numpy as np
import sys, os, pickle
from mpi4py import MPI

import utilities as ut
import setup


def check_edit_param(input_list, param):
	"""
	check_sim_param(input_list, param)

	Checks input_list to overwrite simulation parameters in param dictionary during editor mode
	"""

	if not param: param = get_param_defaults()

	keys = []

	if ('-dt' in input_list): 
		param['dt'] = float(input_list[input_list.index('-dt') + 1])
		keys.append('dt')
	if ('-mass' in input_list): 
		param['mass'] = float(input_list[input_list.index('-mass') + 1])
		keys.append('mass')
	if ('-vdw_sigma' in input_list): 
		param['vdw_sigma'] = float(input_list[input_list.index('-vdw_sigma') + 1])
		keys.append('vdw_sigma')
	if ('-bond_r0' in input_list): 
		param['bond_r0'] = float(input_list[input_list.index('-bond_r0') + 1])
		keys.append('bond_r0')
	if ('-vdw_epsilon' in input_list): 
		param['vdw_epsilon'] = float(input_list[input_list.index('-vdw_epsilon') + 1])
		keys.append('vdw_epsilon')
	if ('-bond_k0' in input_list): 
		param['bond_k0'] = float(input_list[input_list.index('-bond_k0') + 1])
		keys.append('bond_k0')
	if ('-angle_k0' in input_list): 
		param['angle_k0'] = float(input_list[input_list.index('-angle_k0') + 1])
		keys.append('angle_k0')
	if ('-rc' in input_list): 
		param['rc'] = float(input_list[input_list.index('-rc') + 1])
		keys.append('rc')
	if ('-kBT' in input_list): 
		param['kBT'] = float(input_list[input_list.index('-kBT') + 1])
		keys.append('kBT')
	if ('-gamma' in input_list): 
		param['gamma'] = float(input_list[input_list.index('-gamma') + 1])
		keys.append('gamma')
	param['sigma'] = np.sqrt(param['gamma'] * (2 - param['gamma']) * (param['kBT'] / param['mass']))
	if ('-density' in input_list): 
		param['density'] = float(input_list[input_list.index('-density') + 1])
		keys.append('density')
	if ('-save_step' in input_list): 
		param['save_step'] = int(input_list[input_list.index('-save_step') + 1])
		keys.append('save_step')

	return param, keys


def repeat_pos_array(pos, vel, cell_dim, param, n_rep_x=1, n_rep_y=1, n_rep_z=1):
	"""
	create_pos_array(param)

	Form initial positional array of beads

	Parameters
	----------
	
	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions

	param:  dict
		Dictionary of simulation and analysis parameters

	n_rep_x:  int
		Number of repetitions in x direction

	n_rep_y:  int
		Number of repetitions in y direction

	n_rep_z:  int
		Number of repetitions in z direction

		
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions

	param:  dict
		Dictionary of simulation and analysis parameters

	"""

	n_rep_tot = n_rep_x * n_rep_y * n_rep_z

	rep_pos = np.zeros((param['n_bead'] * n_rep_tot, param['n_dim']), dtype=float)
	rep_vel = np.zeros((param['n_bead'] * n_rep_tot, param['n_dim']), dtype=float)

	bond_matrix = np.zeros((param['n_bead']* n_rep_tot, param['n_bead']* n_rep_tot))
	vdw_matrix = np.ones((param['n_bead']* n_rep_tot, param['n_bead']* n_rep_tot)) - np.identity(param['n_bead']* n_rep_tot)

	indices_bond = np.where(param['bond_matrix'])
	indices_vdw = np.where(param['vdw_matrix'] > 1)
	for i in range(n_rep_tot): 
		new_indices_bond = (indices_bond[0] + (param['n_bead'] * i), indices_bond[1] + (param['n_bead'] * i))
		new_indices_vdw = (indices_vdw[0] + (param['n_bead'] * i), indices_vdw[1] + (param['n_bead'] * i))
		bond_matrix[new_indices_bond] = param['bond_matrix'][indices_bond]
		vdw_matrix[new_indices_vdw] = param['vdw_matrix'][indices_vdw]

	print(" Creating simulation cell containing {} fibrils".format(param['n_fibril'] * n_rep_tot))

	bead_list = np.arange(0, param['n_bead'])

	if param['n_dim'] == 2:

		size_x = cell_dim[0] 
		size_y = cell_dim[1] 

		for i in range(n_rep_x):
			for j in range(n_rep_y):
			
				cell = (j + i * n_rep_y)

				pos_x = pos.T[0][bead_list] + size_x * i
				pos_y = pos.T[1][bead_list] + size_y * j

				rep_pos[bead_list + param['n_bead'] * cell] += np.array((pos_x, pos_y)).T
				rep_vel[bead_list + param['n_bead'] * cell] += vel

		cell_dim *= np.array([n_rep_x, n_rep_y])

	elif param['n_dim'] == 3:
		size_x = cell_dim[0] 
		size_y = cell_dim[1]
		size_z = cell_dim[2]

		for k in range(n_rep_z):
			for i in range(n_rep_x):
				for j in range(n_rep_y):
				
					cell = (j + i * n_rep_y + k * n_rep_x * n_rep_y)

					pos_x = pos.T[0][bead_list] + size_x * i
					pos_y = pos.T[1][bead_list] + size_y * j
					pos_z = pos.T[2][bead_list] + size_z * k

					rep_pos[bead_list + param['n_bead'] * cell] += np.array((pos_x, pos_y, pos_z)).T
					rep_vel[bead_list + param['n_bead'] * cell] += vel

		cell_dim *= np.array([n_rep_x, n_rep_y, n_rep_z])

	param['n_fibril_x'] = int(param['n_fibril_x'] * n_rep_x)
	param['n_fibril_y'] = int(param['n_fibril_y'] * n_rep_y)
	param['n_fibril_z'] = int(param['n_fibril_z'] * n_rep_z)
	param['n_fibril'] = int(param['n_fibril'] * n_rep_tot)
	param['n_bead'] = int(param['n_bead'] * n_rep_tot)
	param['bond_matrix'] = bond_matrix
	param['vdw_matrix'] = vdw_matrix

	print(" New Simulation Parameters:")
	keys = ['n_fibril_x', 'n_fibril_y', 'n_fibril_z', 'n_fibril', 'n_bead'] 
	for key in keys: print(" {:<15s} : {}".format(key, param[key]))	

	return rep_pos, rep_vel, cell_dim, param


def editor(current_dir, input_file_name=False):	

	print("\n Entering Editor\n")

	sim_dir = current_dir + '/sim/'

	file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name)
	print("\n Loading restart file {}.npy\n".format(sim_dir + file_names['restart_file_name']))
	restart = ut.load_npy(sim_dir + file_names['restart_file_name'])
	pos = restart[0]
	vel = restart[1]
	cell_dim = pos[-1]
	pos = pos[:-1]

	if ('-nrepx' in sys.argv): n_rep_x = int(sys.argv[sys.argv.index('-nrepx') + 1]) + 1
	else: n_rep_x = 1
	if ('-nrepy' in sys.argv): n_rep_y = int(sys.argv[sys.argv.index('-nrepy') + 1]) + 1
	else: n_rep_y = 1
	if ('-nrepz' in sys.argv): n_rep_z = int(sys.argv[sys.argv.index('-nrepz') + 1]) + 1
	else: n_rep_z = 1

	param, keys = check_edit_param(sys.argv, param)

	if (n_rep_x * n_rep_y * n_rep_z) > 1:
		run_temp = True
		keys += ['n_fibril_x', 'n_fibril_y', 'n_fibril_z', 'n_fibril', 'n_bead'] 
		pos, vel, cell_dim, param = repeat_pos_array(pos, vel, cell_dim, param, n_rep_x, n_rep_y, n_rep_z)
	else: run_temp = False

	print(" New Simulation Parameters:")
	for key in keys: print(" {:<15s} : {}".format(key, param[key]))

	if run_temp:
		from simulation import equilibrate_temperature 
		pos, vel = equilibrate_temperature(sim_dir, pos, cell_dim, param['bond_matrix'], param['vdw_matrix'], param)

	print("\n Saving parameter file {}".format(file_names['param_file_name']))
	pickle.dump(param, open(sim_dir + file_names['param_file_name'] + '.pkl', 'wb'))

	print(" Saving restart file {}\n".format(file_names['restart_file_name']))
	ut.save_npy(sim_dir + file_names['restart_file_name'], (np.vstack((pos, cell_dim)), vel))


