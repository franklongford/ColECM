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
	
	defaults = {	'n_dim' : 2,
		    	'dt' : 0.004,
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
			'n_fibre_x' : 2,
			'n_fibre_y' : 2,
			'n_fibre_z' : 1,
			'n_fibre' : 4,
			'l_fibre' : 5,
			'n_bead' : 20,
			'l_conv' : 1.,
			'res' : 7.5,
			'sharp' : 3.0,
			'skip' : 1,
			'P_0' : 1,
			'lambda_p' : 2E-5}

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

	if ('-ndim' in input_list):
		param['n_dim'] = int(input_list[input_list.index('-ndim') + 1])
		if param['n_dim'] == 2: param['dt'] = 0.004 
		elif param['n_dim'] == 3: param['dt'] = 0.003 
	if ('-mass' in input_list): param['mass'] = float(input_list[input_list.index('-mass') + 1])
	if ('-vdw_sigma' in input_list): param['vdw_sigma'] = float(input_list[input_list.index('-vdw_sigma') + 1])
	param['bond_r0'] = 2.**(1./6.) * param['vdw_sigma']
	if ('-vdw_epsilon' in input_list): param['vdw_epsilon'] = float(input_list[input_list.index('-vdw_epsilon') + 1])
	if ('-bond_k' in input_list): param['bond_k'] = float(input_list[input_list.index('-bond_k') + 1])
	if ('-angle_k' in input_list): param['angle_k'] = float(input_list[input_list.index('-angle_k') + 1])
	if ('-rc' in input_list): param['rc'] = float(input_list[input_list.index('-rc') + 1])
	else: param['rc'] = param['vdw_sigma'] * 3.0
	if ('-kBT' in input_list): param['kBT'] = float(input_list[input_list.index('-kBT') + 1])
	if ('-gamma' in input_list): param['gamma'] = float(input_list[input_list.index('-gamma') + 1])
	param['sigma'] = np.sqrt(param['gamma'] * (2 - param['gamma']) * (param['kBT'] / param['mass']))
	if ('-nfibx' in input_list): param['n_fibre_x'] = int(input_list[input_list.index('-nfibx') + 1])
	if ('-nfiby' in input_list): param['n_fibre_y'] = int(input_list[input_list.index('-nfiby') + 1])
	if ('-nfibz' in input_list): param['n_fibre_z'] = int(input_list[input_list.index('-nfibz') + 1])
	param['n_fibre'] = param['n_fibre_x'] * param['n_fibre_y']
	if param['n_dim'] == 3: param['n_fibre'] *= param['n_fibre_z']
	if ('-lfib' in input_list): param['l_fibre'] = int(input_list[input_list.index('-lfib') + 1])
	param['n_bead'] = param['n_fibre'] * param['l_fibre']

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
	param['l_fibre']  = int(input("Enter length of fibril (no. of beads): "))
	param['n_fibre_x']  = int(input("Enter number of fibrils in x dimension: "))
	param['n_fibre_y']  = int(input("Enter number of fibrils in y dimension: "))
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

	keys = ['n_dim', 'dt', 'mass', 'vdw_sigma', 'vdw_epsilon', 'bond_r0', 'bond_k', 'angle_theta0', 'angle_k', 'rc', 'kBT', 
		'gamma', 'sigma', 'l_fibre', 'n_fibre_x', 'n_fibre_y', 'n_fibre_z', 'n_fibre', 'n_bead']

	if os.path.exists(sim_dir + file_names['param_file_name'] + '.pkl'):
		print(" Loading parameter file {}.pkl".format(sim_dir + file_names['param_file_name']))
		param_file = ut.read_param_file(sim_dir + file_names['param_file_name'])
		keys += ['bond_matrix', 'vdw_matrix', 'l_conv']
		for key in keys: param[key] = param_file[key]		

	else:
		if input_file_name: _, param = read_input_file(input_file_name, simulation=True, param=param)
		param = check_sim_param(sys.argv, param)

		print(" Creating parameter file {}.pkl".format(sim_dir + file_names['param_file_name'])) 
		ut.make_param_file(sim_dir + file_names['param_file_name'])

		for key in keys: ut.update_param_file(sim_dir + file_names['param_file_name'], key, param[key])
		
	assert param['n_dim'] in [2, 3]
	assert param['rc'] >= 1.5 * param['vdw_sigma']

	if ('-nstep' in sys.argv): param['n_step'] = int(sys.argv[sys.argv.index('-nstep') + 1])
	if ('-save_step' in sys.argv): param['save_step'] = int(sys.argv[sys.argv.index('-save_step') + 1])

	param = check_analysis_param(sys.argv, param)	

	return file_names, param


def grow_fibre(index, bead, pos, param, bond_matrix, vdw_matrix, max_energy=100, max_attempt=200):
	"""
	grow_fibre(index, bead, pos, n_bead, param, max_energy, max_attempt=200)

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

	max_attempt:  int
		Maximum number of attempts to find an acceptable configuration
		
		
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Updated positions of n_bead beads in n_dim

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Updated matrix determining whether a bond is present between two beads

	"""

	if param['n_dim'] == 2: from sim_tools_2D import calc_energy_forces
	elif param['n_dim'] == 3: from sim_tools_3D import calc_energy_forces

	cell_dim = np.array([param['vdw_sigma']**2 * param['n_bead']] * param['n_dim'])

	if bead == 0:
		pos[index] = np.random.random((param['n_dim'])) * param['vdw_sigma'] * 2

	else:
		bond_beads, dist_index, r_index = ut.update_bond_lists(bond_matrix)

		energy = max_energy + 1
		attempt = 0

		while energy > max_energy:
			new_vec = ut.rand_vector(param['n_dim']) * param['vdw_sigma']	
			pos[index] = pos[index-1] + new_vec
			distances = ut.get_distances(pos[:bead+1], cell_dim)
			r2 = np.sum(distances**2, axis=0)

			energy, _, _ = calc_energy_forces(distances, r2, param, bond_matrix, vdw_matrix, ut.check_cutoff(r2, param['rc']**2), bond_beads, dist_index, r_index)

			attempt += 1
			if attempt > max_attempt: raise RuntimeError

	return pos


def create_pos_array(param):
	"""
	create_pos_array(param)

	Form initial positional array of beads

	Parameters
	----------

	n_dim:  int
		Number of dimensions in simulation

	n_fibril_x:  int
		Number of fibrils in x dimension

	n_fibril_y:  int
		Number of fibrils in y dimension

	l_fibril:  int
		Length of each fibre in simulation

	vdw_param: array_like (float); shape=(2)
		Sigma and epsilon paameters for Van de Waals forces

	bond_param: array_like (float); shape=(2)
		Equilibrium length and energy paameters for bonded forces

	angle_param: array_like (float); shape=(2)
		Equilibrium angle and energy paameters for angular forces

	rc:  float
		Interaction cutoff radius for non-bonded forces
		
		
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	"""

	pos = np.zeros((param['n_bead'], param['n_dim']), dtype=float)
	bond_matrix = np.zeros((param['n_bead'], param['n_bead']), dtype=int)
	vdw_matrix = np.zeros(param['n_bead'], dtype=int)

	for bead in range(param['n_bead']):
		if bead % param['l_fibre'] == 0: vdw_matrix[bead] += 10
		elif bead % param['l_fibre'] == param['l_fibre']-1: vdw_matrix[bead] += 10
		else: vdw_matrix[bead] += 1

	vdw_matrix = np.reshape(np.tile(vdw_matrix, (1, param['n_bead'])), (param['n_bead'], param['n_bead']))

	for bead in range(param['n_bead']): vdw_matrix[bead][bead] = 0

	for fibre in range(param['n_fibre']):
		for bead in range(1, param['l_fibre']):
			n = fibre * param['l_fibre'] + bead
			bond_matrix[n][n-1] = 1
			bond_matrix[n-1][n] = 1

	print(" Creating fibre template containing {} beads".format(param['l_fibre']))

	init_pos = np.zeros((param['l_fibre'], param['n_dim']), dtype=float)
	bead = 0

	while bead < param['l_fibre']:
		try:
			init_pos = grow_fibre(bead, bead, init_pos, param,
						bond_matrix[[slice(0, bead+1) for _ in bond_matrix.shape]],
						vdw_matrix[[slice(0, bead+1) for _ in vdw_matrix.shape]])
			bead += 1
		except RuntimeError: bead = 0

	pos[range(param['l_fibre'])] += init_pos
	pos -= np.min(pos)

	print(" Creating simulation cell containing {} fibres".format(param['n_fibre']))

	if param['n_dim'] == 2:

		size_x = np.max(pos.T[0]) + 2 * param['vdw_sigma'] 
		size_y = np.max(pos.T[1]) + 2 * param['vdw_sigma'] 
		bead_list = np.arange(0, param['l_fibre'])

		for i in range(param['n_fibre_x']):
			for j in range(param['n_fibre_y']):
				if j + i == 0: continue
			
				fibre = (j + i * param['n_fibre_y'])

				pos_x = pos.T[0][bead_list] + size_x * i
				pos_y = pos.T[1][bead_list] + size_y * j

				pos[bead_list + param['l_fibre'] * fibre] += np.array((pos_x, pos_y)).T

		cell_dim = np.array([np.max(pos.T[0]) + param['vdw_sigma'], np.max(pos.T[1]) + param['vdw_sigma']])

	elif param['n_dim'] == 3:
		size_x = np.max(pos.T[0]) + param['vdw_sigma'] / 2
		size_y = np.max(pos.T[1]) + param['vdw_sigma'] / 2
		size_z = np.max(pos.T[2]) + param['vdw_sigma'] / 2
		bead_list = np.arange(0, param['l_fibre'])

		for k in range(param['n_fibre_z']):
			for i in range(param['n_fibre_x']):
				for j in range(param['n_fibre_y']):
					if k + j + i == 0: continue
				
					fibre = (j + i * param['n_fibre_y'] + k * param['n_fibre_x'] * param['n_fibre_y'])

					pos_x = pos.T[0][bead_list] + size_x * i
					pos_y = pos.T[1][bead_list] + size_y * j
					pos_z = pos.T[2][bead_list] + size_z * k

					pos[bead_list + param['l_fibre'] * fibre] += np.array((pos_x, pos_y, pos_z)).T

		cell_dim = np.array([np.max(pos.T[0]) + param['vdw_sigma'] / 2, np.max(pos.T[1]) + param['vdw_sigma'] / 2, np.max(pos.T[2]) + param['vdw_sigma'] / 2])

	return pos, cell_dim, bond_matrix, vdw_matrix


def equilibrate_pressure(pos, vel, cell_dim, bond_matrix, vdw_matrix, param, thresh=2E-2):

	print("\n" + " " * 15 + "----Equilibrating Pressure----\n")

	if param['n_dim'] == 2: from sim_tools_2D import velocity_verlet_alg
	elif param['n_dim'] == 3: from sim_tools_3D import velocity_verlet_alg

	sqrt_dt = np.sqrt(param['dt'])

	frc, verlet_list, pot_energy, virial_tensor, verlet_list, bond_beads, dist_index, r_index = calc_state(pos, vel, cell_dim, bond_matrix, vdw_matrix, param)

	kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
	P = 1. / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
	P_array = [P]
	step = 1
	optimising = True

	print(" Starting pressure:  {:>10.4f}\n Reference pressure: {:>10.4f}".format(P, param['P_0']))
	print(" Starting volume:    {:>10.4f}\n".format(np.prod(cell_dim)))
	print(" {:^18s} | {:^18s} | {:^18s} ".format('Step', 'Pressure', 'Volume'))
	print(" " + "-" * 60)

	while optimising:
		pos, vel, frc, cell_dim, verlet_list, pot_energy, virial_tensor = velocity_verlet_alg(pos, vel, frc, virial_tensor, param, bond_matrix, vdw_matrix, 
			verlet_list, bond_beads, dist_index, r_index, param['dt'], sqrt_dt, cell_dim, NPT=True)

		kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
		P = 1. / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
		P_array.append(P)

		if step % 2000 == 0: 
			av_P = np.mean(P_array)
			optimising = (abs(av_P - param['P_0']) >= thresh)
			P_array = [P]
			print(" {:18d} | {:>18.4f} | {:>18.4f} ".format(step, av_P, np.prod(cell_dim)))
 
		step += 1

	print("\n No. iterations:   {:>10d}".format(step))
	print(" Final pressure:   {:>10.4f}".format(P))
	print(" Final volume:     {:>10.4f}".format(np.prod(cell_dim)))

	return pos, vel, cell_dim


def equilibrate_temperature(pos, vel, cell_dim, bond_matrix, vdw_matrix, param, thresh=5E-2):

	print("\n" + " " * 15 + "----Equilibrating Temperature----\n")

	if param['n_dim'] == 2: from sim_tools_2D import velocity_verlet_alg
	elif param['n_dim'] == 3: from sim_tools_3D import velocity_verlet_alg

	sqrt_dt = np.sqrt(param['dt'])
	n_dof = param['n_dim'] * (param['n_bead'] - 1) 

	frc, verlet_list, pot_energy, virial_tensor, verlet_list, bond_beads, dist_index, r_index = calc_state(pos, vel, cell_dim, bond_matrix, vdw_matrix, param)

	kBT = 2 * ut.kin_energy(vel, param['mass'], param['n_dim'])
	step = 1
	kBT_array = [kBT]
	optimising = True

	print(" Starting kBT:    {:>10.4f}\n Reference kBT:   {:>10.4f}".format(kBT, param['kBT']))
	print(" {:^18s} | {:^18s} ".format('Step', 'kBT'))
	print(" " + "-" * 40)

	while optimising:
		pos, vel, frc, cell_dim, verlet_list, pot_energy, virial_tensor = velocity_verlet_alg(pos, vel, frc, virial_tensor, param, bond_matrix, vdw_matrix, 
			verlet_list, bond_beads, dist_index, r_index, param['dt'], sqrt_dt, cell_dim)

		kBT = 2 * ut.kin_energy(vel, param['mass'], param['n_dim']) / n_dof
		kBT_array.append(kBT)

		if step % 1000 == 0: 
			av_kBT = np.mean(kBT_array)
			optimising = (abs(av_kBT - param['kBT']) >= thresh)
			kBT_array = [kBT]
			print(" {:18d} | {:>18.4f} ".format(step, av_kBT))
 
		step += 1

	print("\n No. iterations:   {:>10d}".format(step))
	print(" Final kBT:   {:>10.4f}".format(kBT))

	return pos, vel



def calc_state(pos, vel, cell_dim, bond_matrix, vdw_matrix, param):
	"""
	calc_state(pos, vel, cell_dim, bond_matrix, vdw_matrix, param)
	
	Calculate state of simulation using starting configuration and parameters provided

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim
	
	vel: array_like, dtype=float
		Velocity of each bead in all collagen fibres

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	param:  

	
	Returns
	-------

	frc: array_like, dtype=float
		Forces acting upon each bead in all collagen fibres

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance

	pot_energy:  float
		Total potential energy of system

	virial_tensor:  array_like, (float); shape=(n_dim, n_dim)
		Virial components of pressure tensor of system

	bond_beads:  array_like, (int); shape=(n_angle, 3)
		Array containing indicies in pos array all 3-bead angular interactions

	dxyz_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in dx and dy arrays of all bonded interactions

	r_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in r array of all bonded interactions
	
	"""

	if param['n_dim'] == 2: from sim_tools_2D import calc_energy_forces
	elif param['n_dim'] == 3: from sim_tools_3D import calc_energy_forces

	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)

	verlet_list = ut.check_cutoff(r2, param['rc']**2)

	bond_beads, dist_index, r_index = ut.update_bond_lists(bond_matrix)
	pot_energy, frc, virial_tensor = calc_energy_forces(distances, r2, param, bond_matrix, vdw_matrix, verlet_list, bond_beads, dist_index, r_index)

	return frc, verlet_list, pot_energy, virial_tensor, verlet_list, bond_beads, dist_index, r_index


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

	if os.path.exists(sim_dir + file_names['restart_file_name'] + '.npy'):
		print(" Loading restart file {}.npy".format(sim_dir + file_names['restart_file_name']))
		restart = ut.load_npy(sim_dir + file_names['restart_file_name'])
		pos = restart[0]
		vel = restart[1]
		cell_dim = pos[-1]
		pos = pos[:-1]

		bond_matrix = param['bond_matrix'] 
		vdw_matrix = param['vdw_matrix'] 

	elif os.path.exists(sim_dir + file_names['pos_file_name'] + '.npy'):
		print(" Loading position file {}.npy".format(sim_dir + file_names['pos_file_name']))
		pos = ut.load_npy(sim_dir + file_names['pos_file_name'])
		cell_dim = pos[-1]
		pos = pos[:-1]
		vel = (np.random.random(pos.shape) - 0.5) * np.sqrt(2 * param['kBT'] / param['mass'])

		bond_matrix = param['bond_matrix'] 
		vdw_matrix = param['vdw_matrix'] 

	else:
		file_names['pos_file_name'] = ut.check_file_name(file_names['pos_file_name'], file_type='pos') + '_pos'

		print(" Creating input pos file {}{}.npy".format(sim_dir, file_names['pos_file_name']))

		pos, cell_dim, bond_matrix, vdw_matrix = create_pos_array(param)
		vel = np.random.normal(loc=np.sqrt(param['kBT'] / param['mass']), size=pos.shape) 

		param['bond_matrix'] = bond_matrix
		param['vdw_matrix'] = vdw_matrix
		param['l_conv'] = 1 / (2 * param['vdw_sigma'])

		keys = ['bond_matrix', 'vdw_matrix', 'l_conv']
		for key in keys: ut.update_param_file(sim_dir + file_names['param_file_name'], key, param[key])

		print(" Saving input pos file {}{}.npy".format(sim_dir, file_names['pos_file_name']))
		ut.save_npy(sim_dir + file_names['pos_file_name'], np.vstack((pos, cell_dim)))

		pos, vel, cell_dim = equilibrate_pressure(pos, vel, cell_dim, bond_matrix, vdw_matrix, param)
		pos, vel = equilibrate_temperature(pos, vel, cell_dim, bond_matrix, vdw_matrix, param)

		print(" Saving restart file {}".format(file_names['restart_file_name']))
		ut.save_npy(sim_dir + file_names['restart_file_name'], (np.vstack((pos, cell_dim)), vel))

	return pos, vel, cell_dim, param



