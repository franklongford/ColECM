"""
ColECM: Collagen ExtraCellular Matrix Simulation
SETUP ROUTINE 

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 12/04/2018
"""

import numpy as np
import scipy as sp
import random

import sys, os, pickle

import utilities as ut


def import_files(n_dim, param_file_name, pos_file_name, restart_file_name):

	if n_dim == 2: from sim_tools_2D import create_pos_array
	elif n_dim == 3: from sim_tools_3D import create_pos_array

	if not os.path.exists('{}.pkl'.format(param_file_name)):
		print("Creating parameter file {}.pkl".format(param_file_name)) 
		ut.make_param_file(param_file_name)
	else:
		print("Loading parameter file {}.pkl".format(param_file_name))
		param_file = ut.read_param_file(param_file_name)

	try: mass = param_file['mass']
	except:
		if ('-mass' in sys.argv): mass = float(sys.argv[sys.argv.index('-mass') + 1])
		else: mass = float(input("Enter bead mass: "))
		param_file = ut.update_param_file(param_file_name, 'mass', mass)

	try: vdw_param = param_file['vdw_param']
	except:
		if ('-vdw_sigma' in sys.argv): vdw_sigma = float(sys.argv[sys.argv.index('-vdw_sigma') + 1])
		else: vdw_sigma = float(input("Enter vdw sigma radius: "))
		if ('-vdw_epsilon' in sys.argv): vdw_epsilon = float(sys.argv[sys.argv.index('-vdw_epsilon') + 1])
		else: vdw_epsilon = float(input("Enter vdw epsilon energy: "))
		vdw_param = [vdw_sigma, vdw_epsilon]
		param_file = ut.update_param_file(param_file_name, 'vdw_param', vdw_param)

	try: bond_param = param_file['bond_param']
	except:
		bond_r0 = 2.**(1./6.) * vdw_param[0]
		if ('-bond_k' in sys.argv): bond_k = float(sys.argv[sys.argv.index('-bond_k') + 1])
		else: bond_k = float(input("Enter bond k energy: "))
		bond_param = [bond_r0, bond_k]
		param_file = ut.update_param_file(param_file_name, 'bond_param', bond_param)

	try: angle_param = param_file['angle_param']
	except:
		angle_theta0 = np.pi
		if ('-angle_k' in sys.argv): angle_k = float(sys.argv[sys.argv.index('-angle_k') + 1])
		else: angle_k = float(input("Enter angle k energy: "))
		angle_param = [angle_theta0, angle_k]
		param_file = ut.update_param_file(param_file_name, 'angle_param', angle_param)

	try: rc = param_file['rc']
	except: 
		rc = 3.25 * vdw_param[0]
		param_file = ut.update_param_file(param_file_name, 'rc', rc)

	try: Langevin = param_file['Langevin']
	except:
		if ('-Langevin' in sys.argv): Langevin = bool(sys.argv[sys.argv.index('-Langevin') + 1].upper() == 'Y')
		else: Langevin = bool(input("Langevin thermostat? (Y/N) ").upper() == 'Y')
		param_file = ut.update_param_file(param_file_name, 'Langevin', Langevin)

	if Langevin:

		try: kBT = param_file['kBT']
		except: 
			if ('-kBT' in sys.argv): kBT = float(sys.argv[sys.argv.index('-kBT') + 1])
			else: kBT = float(input("Enter kBT constant: "))
			param_file = ut.update_param_file(param_file_name, 'kBT', kBT)

		try: thermo_gamma = param_file['thermo_gamma']
		except:
			if ('-gamma' in sys.argv): thermo_gamma = float(sys.argv[sys.argv.index('-gamma') + 1])
			else: thermo_gamma = float(input("Enter Langevin gamma constant: "))
			param_file = ut.update_param_file(param_file_name, 'thermo_gamma', thermo_gamma)

	else: 
		kBT = 1.0
		param_file = ut.update_param_file(param_file_name, 'kBT', kBT)
		thermo_gamma = 0.0
		param_file = ut.update_param_file(param_file_name, 'thermo_gamma', thermo_gamma)

	params = (mass, vdw_param, bond_param, angle_param, rc, Langevin, kBT, thermo_gamma)

	if os.path.exists(restart_file_name + '.npy'):
		print("Loading restart file {}.npy".format(restart_file_name))
		restart = ut.load_npy(restart_file_name)
		pos = restart[0]
		vel = restart[1]
		cell_dim = pos[-1]
		pos = pos[:-1]

		bond_matrix = param_file['bond_matrix']
		vdw_matrix = param_file['vdw_matrix']
		l_conv = param_file['l_conv']

	elif os.path.exists(pos_file_name + '.npy'):
		print("Loading position file {}.npy".format(pos_file_name))
		pos = ut.load_npy(pos_file_name)
		cell_dim = pos[-1]
		pos = pos[:-1]

		bond_matrix = param_file['bond_matrix']
		vdw_matrix = param_file['vdw_matrix']
		l_conv = param_file['l_conv']

		vel = (np.random.random(pos.shape) - 0.5) * np.sqrt(2 * kBT / mass)
		
	else:
		pos_file_name = ut.check_file_name(pos_file_name, file_type='pos') + '_pos'

		print("Creating input pos file {}.npy".format(pos_file_name))

		if ('-nfibx' in sys.argv): n_fibril_x = int(sys.argv[sys.argv.index('-nfibx') + 1])
		else: n_fibril_x = int(input("Enter number of fibrils in x dimension: "))
		if ('-nfiby' in sys.argv): n_fibril_y = int(sys.argv[sys.argv.index('-nfiby') + 1])
		else: n_fibril_y = int(input("Enter number of fibrils in y dimension: "))
		if ('-lfib' in sys.argv): l_fibril = int(sys.argv[sys.argv.index('-lfib') + 1])
		else: l_fibril = int(input("Enter length of fibril (no. of beads): "))

		l_conv = 10. / (l_fibril * 2 * vdw_param[0])

		pos, cell_dim, bond_matrix, vdw_matrix = create_pos_array(n_dim, n_fibril_x, n_fibril_y, l_fibril, vdw_param, bond_param, angle_param, rc)
		print("Saving input pos file {}.npy".format(pos_file_name))

		ut.save_npy(pos_file_name, np.vstack((pos, cell_dim)))

		param_file = ut.update_param_file(param_file_name, 'bond_matrix', bond_matrix)
		param_file = ut.update_param_file(param_file_name, 'vdw_matrix', vdw_matrix)
		param_file = ut.update_param_file(param_file_name, 'l_conv', l_conv)

		vel = (np.random.random(pos.shape) - 0.5) * np.sqrt(2 * kBT / mass)
		

	return pos, vel, cell_dim, l_conv, bond_matrix, vdw_matrix, params


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


def initial_state(n_dim, pos, cell_dim, bond_matrix, vdw_matrix, vdw_param, bond_param, angle_param, rc, kBT=1.0):
	"""
	setup(n_dim, cell_dim, nchain, lchain, mass, kBT, vdw_param, bond_param, angle_param, rc)
	
	Setup simulation using parameters provided

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

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

	if n_dim == 2: from sim_tools_2D import calc_energy_forces
	elif n_dim == 3: from sim_tools_3D import calc_energy_forces

	n_bead = pos.shape[0]
	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)

	verlet_list = ut.check_cutoff(r2, rc**2)

	bond_beads, dxy_index, r_index = ut.update_bond_lists(bond_matrix)
	_, frc = calc_energy_forces(distances, r2, bond_matrix, vdw_matrix, verlet_list, 
				vdw_param, bond_param, angle_param, rc, bond_beads, dxy_index, r_index)

	return frc, verlet_list, bond_beads, dxy_index, r_index
