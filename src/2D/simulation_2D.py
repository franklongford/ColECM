"""
COLLAGEN FIBRE SIMULATION 2D

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 09/03/2018
"""

import numpy as np
import random

import sys, os, pickle

import utilities_2D as ut

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)


def import_files(n_dim, param_file_name, pos_file_name):

	params = ()

	if not os.path.exists('{}.pkl'.format(param_file_name)):
		print("Creating parameter file {}.pkl".format(param_file_name)) 
		ut.make_param_file(param_file_name)
	else:
		print("Loading parameter file {}.pkl".format(param_file_name))
		param_file = ut.read_param_file(param_file_name)

	try: mass = param_file['mass']
	except:
		mass = float(input("Enter bead mass: "))
		param_file = ut.update_param_file(param_file_name, 'mass', mass)

	try: vdw_param = param_file['vdw_param']
	except:
		vdw_sigma = float(input("Enter vdw sigma radius: "))
		vdw_epsilon = float(input("Enter vdw epsilon energy: "))
		vdw_param = [vdw_sigma, vdw_epsilon]
		param_file = ut.update_param_file(param_file_name, 'vdw_param', vdw_param)

	try: bond_param = param_file['bond_param']
	except:
		bond_r0 = 2.**(1./6.) * vdw_sigma
		bond_k = float(input("Enter bond k energy: "))
		bond_param = [bond_r0, bond_k]
		param_file = ut.update_param_file(param_file_name, 'bond_param', bond_param)

	try: angle_param = param_file['angle_param']
	except:
		angle_theta0 = np.pi
		angle_k = float(input("Enter angle k energy: "))
		angle_param = [angle_theta0, angle_k]
		param_file = ut.update_param_file(param_file_name, 'angle_param', angle_param)

	try: rc = param_file['rc']
	except: 
		rc = 4 * vdw_sigma
		param_file = ut.update_param_file(param_file_name, 'rc', rc)

	try: kBT = param_file['kBT']
	except: 
		kBT = float(input("Enter kBT constant: "))
		param_file = ut.update_param_file(param_file_name, 'kBT', kBT)

	try: Langevin = param_file['Langevin']
	except: 
		Langevin = bool(input("Langevin thermostat? (Y/N) ").upper() == 'Y')
		param_file = ut.update_param_file(param_file_name, 'Langevin', Langevin)

	params = (mass, vdw_param, bond_param, angle_param, rc, kBT, Langevin)

	if Langevin: 
		try: thermo_gamma = param_file['thermo_gamma']
		except: 
			thermo_gamma = float(input("Enter Langevin gamma constant: "))
			param_file = ut.update_param_file(param_file_name, 'thermo_gamma', thermo_gamma)

		try: thermo_sigma = param_file['thermo_sigma']
		except: 
			thermo_sigma =  np.sqrt(2 * kBT * thermo_gamma / mass)
			param_file = ut.update_param_file(param_file_name, 'thermo_sigma', thermo_sigma)

		params += (thermo_gamma, thermo_sigma)


	if not os.path.exists(pos_file_name + '.npy'):

		print("Creating input pos file {}.npy".format(pos_file_name))

		n_fibre = int(input("Enter square root of number of fibrils: "))
		n_fibre *= n_fibre
		l_fibre = int(input("Enter length of fibril (no. of beads): "))

		l_conv = 10. / (l_fibre * 2 * vdw_param[0])

		pos, cell_dim, bond_matrix, vdw_matrix = create_pos_array(n_dim, n_fibre, l_fibre, vdw_param, bond_param, angle_param, rc)
		print("Saving input pos file {}.npy".format(pos_file_name))
		ut.save_npy(pos_file_name, pos)

		param_file = ut.update_param_file(param_file_name, 'cell_dim', cell_dim)
		param_file = ut.update_param_file(param_file_name, 'bond_matrix', bond_matrix)
		param_file = ut.update_param_file(param_file_name, 'vdw_matrix', vdw_matrix)
		param_file = ut.update_param_file(param_file_name, 'l_conv', l_conv)
		
	else:

		print("Loading input pos file {}.npy".format(pos_file_name))
		pos = ut.load_npy(pos_file_name)
		cell_dim = param_file['cell_dim']
		bond_matrix = param_file['bond_matrix']
		vdw_matrix = param_file['vdw_matrix']
		l_conv = param_file['l_conv']

	return pos, cell_dim, l_conv, bond_matrix, vdw_matrix, params



def check_cutoff(array, thresh):
	"""
	check_cutoff(array, rc)

	Determines whether elements of array are less than or equal to thresh
	"""

	return (array <= thresh).astype(float)


def get_dx_dy(pos, cell_dim):
	"""
	get_dx_dy(pos, n_dim, cell_dim)

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

	dx = np.tile(temp_pos[0], (n_bead, 1))
	dy = np.tile(temp_pos[1], (n_bead, 1))

	dx = dx.T - dx
	dy = dy.T - dy

	dx -= cell_dim[0] * np.array(2 * dx / cell_dim[0], dtype=int)
	dy -= cell_dim[1] * np.array(2 * dy / cell_dim[1], dtype=int)

	"""	
	dxyz = np.array([np.tile(temp_pos[0], (n_bead, 1)), np.tile(temp_pos[1], (n_bead, 1))])
	dxyz = np.reshape(np.tile(temp_pos, (1, n_bead)), (n_dim, n_bead, n_bead))
	dxyz = np.transpose(dxyz, axes=(0, 2, 1)) - dxyz
	for i in range(n_dim): dxyz[i] -= cell_dim[i] * np.array(2 * dxyz[i] / cell_dim[i], dtype=int)
	"""	

	return dx, dy


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


def kin_energy(vel):
	"""
	kin_energy(vel)

	Returns kinetic energy of simulation in reduced units
	"""

	return np.mean(vel**2)


def update_bond_lists(bond_matrix):
	"""
	update_bond_lists(bond_matrix)

	Return atom indicies of angular terms
	"""

	N = bond_matrix.shape[0]

	bond_index_half = np.argwhere(np.triu(bond_matrix))
	bond_index_full = np.argwhere(bond_matrix)

	indices_half = ut.create_index(bond_index_half)
	indices_full = ut.create_index(bond_index_full)

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


def cos_sin_theta(vector, r_vector):
	"""
	cos_sin_theta(vector, r_vector)

	Returns cosine and sine of angles of intersecting vectors betwen even and odd indicies

	Parameters
	----------

	vector:  array_like, (float); shape=(n_vector, n_dim)
		Array of displacement vectors between connecting beads

	r_vector: array_like, (float); shape=(n_vector)
		Array of radial distances between connecting beads

	Returns
	-------

	cos_the:  array_like (float); shape=(n_vector/2)
		Cosine of the angle between each pair of displacement vectors

	sin_the: array_like (float); shape=(n_vector/2)
		Sine of the angle between each pair of displacement vectors

	r_prod: array_like (float); shape=(n_vector/2)
		Product of radial distance between each pair of displacement vectors
	"""

	n_vector = int(vector.shape[0])
	temp_vector = np.reshape(vector, (int(n_vector/2), 2, 2))

	"Calculate |rij||rjk| product for each pair of vectors"
	r_prod = np.prod(np.reshape(r_vector, (int(n_vector/2), 2)), axis = 1)

	"Form dot product of each vector pair rij*rjk in vector array corresponding to an angle"
	dot_prod = np.sum(np.prod(temp_vector, axis=1), axis=1)

	"Form pseudo-cross product of each vector pair rij*rjk in vector array corresponding to an angle"
	cross_prod = np.linalg.det(temp_vector)

	"Calculate cos(theta) for each angle"
	cos_the = dot_prod / r_prod

	"Calculate sin(theta) for each angle"
	sin_the = cross_prod / r_prod

	return cos_the, sin_the, r_prod


def calc_energy_forces(dx, dy, r2, bond_matrix, vdw_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxdy_index, r_index):
	"""
	calc_energy_forces(dxy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxdy_index, r_index)

	Return tot potential energy and forces on each bead in simulation

	Parameters
	----------

	dx:  array_like (float); shape=(n_bead, n_bead)
		Displacement along x axis between each bead

	dy:  array_like (float); shape=(n_bead, n_bead)
		Displacement along y axis between each bead

	r2:  array_like (float); shape=(n_bead, n_bead)
		Square of Radial disance between each bead

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

	bond_beads:  array_like, (int); shape=(n_angle, 3)
		Array containing indicies in pos array all 3-bead angular interactions

	dxy_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in dx and dy arrays of all bonded interactions

	r_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in r array of all bonded interactions
		
	Returns
	-------

	pot_energy:  float
		Total potential energy of simulation cell

	frc_beads:  array_like (float); shape=(n_beads, n_dim)
		Forces acting upon each bead due to positional array 

	"""

	n_bead = dx.shape[0]
	#frc_beads = np.zeros((n_dim, n_bead))
	f_beads_x = np.zeros((n_bead))
	f_beads_y = np.zeros((n_bead))
	pot_energy = 0
	cut_frc = force_vdw(rc**2, vdw_param[0], vdw_param[1])
	cut_pot = pot_vdw(rc**2, vdw_param[0], vdw_param[1])
	
	nbond = int(np.sum(np.triu(bond_matrix)))

	if nbond > 0:
		"Bond Lengths"
		bond_index_half = np.argwhere(np.triu(bond_matrix))
		bond_index_full = np.argwhere(bond_matrix)

		indices_half = ut.create_index(bond_index_half)
		indices_full = ut.create_index(bond_index_full)
	
		indices_dxy = ut.create_index(dxdy_index)

		r_half = np.sqrt(r2[indices_half])
		r_full = np.repeat(r_half, 2)
		
		bond_pot = pot_harmonic(r_half, bond_param[0], bond_param[1])
		pot_energy += np.sum(bond_pot)

		bond_frc = force_harmonic(r_half, bond_param[0], bond_param[1])
		for i, sign in enumerate([1, -1]):
			f_beads_x[indices_half[i]] += sign * (bond_frc * dx[indices_half] / r_half)
			f_beads_y[indices_half[i]] += sign * (bond_frc * dy[indices_half] / r_half)

			#frc_beads[0][indices_half[i]] += sign * (bond_frc * dxyz[0][indices_half] / r_half)
			#frc_beads[1][indices_half[i]] += sign * (bond_frc * dxyz[1][indices_half] / r_half)

		"Bond Angles"

		try:
			"Make array of vectors rij, rjk for all connected bonds"
			vector = np.stack((dx[indices_dxy], dy[indices_dxy]), axis=1)
			n_vector = int(vector.shape[0])

			"Find |rij| values for each vector"
			r_vector = r_half[r_index]
			cos_the, sin_the, r_prod = cos_sin_theta(vector, r_vector)
			pot_energy += angle_param[1] * np.sum(cos_the + 1)

			"Form arrays of |rij| vales, cos(theta) and |rij||rjk| terms same shape as vector array"
			r_array = np.reshape(np.repeat(r_vector, 2), vector.shape)
			sin_the_array = np.reshape(np.repeat(sin_the, 4), vector.shape)
			r_prod_array = np.reshape(np.repeat(r_prod, 4), vector.shape)

			"Form left and right hand side terms of (cos(theta) rij / |rij|^2 - rjk / |rij||rjk|)"
			r_left = vector / r_prod_array
			r_right = sin_the_array * vector / r_array**2

			ij_indices = np.arange(0, n_vector, 2)
			jk_indices = np.arange(1, n_vector, 2)

			"Perfrom right hand - left hand term for every rij rkj pair"
			r_left[ij_indices] -= r_right[jk_indices]
			r_left[jk_indices] -= r_right[ij_indices] 
		
			"Calculate forces upon beads i, j and k"
			frc_angle_ij = angle_param[1] * r_left
			frc_angle_k = -np.sum(np.reshape(frc_angle_ij, (int(n_vector/2), 2, 2)), axis=1)

			"Add angular forces to force array" 
			f_beads_x[bond_beads.T[0]] += frc_angle_ij[ij_indices].T[0]
			f_beads_x[bond_beads.T[1]] += frc_angle_k.T[0]
			f_beads_x[bond_beads.T[2]] += frc_angle_ij[jk_indices].T[0]

			f_beads_y[bond_beads.T[0]] += frc_angle_ij[ij_indices].T[1]
			f_beads_y[bond_beads.T[1]] += frc_angle_k.T[1]
			f_beads_y[bond_beads.T[2]] += frc_angle_ij[jk_indices].T[1]

		except IndexError: pass


	non_zero = np.nonzero(r2 * verlet_list)
	nonbond_pot = vdw_matrix[non_zero] * pot_vdw((r2 * verlet_list)[non_zero], vdw_param[0], vdw_param[1]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	nonbond_frc = vdw_matrix[non_zero] * force_vdw((r2 * verlet_list)[non_zero], vdw_param[0], vdw_param[1]) - cut_frc
	temp_x = np.zeros(r2.shape)
	temp_y = np.zeros(r2.shape)
	temp_x[non_zero] += nonbond_frc * (dx[non_zero] / r2[non_zero])
	temp_y[non_zero] += nonbond_frc * (dy[non_zero] / r2[non_zero])

	f_beads_x += np.sum(temp_x, axis=0)
	f_beads_y += np.sum(temp_y, axis=0)

	frc_beads = np.transpose(np.array([f_beads_x, f_beads_y]))

	return pot_energy, frc_beads


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

	cell_dim = np.array([vdw_param[0]**2 * n_bead] * n_dim)

	if bead == 0:
		pos[n] = np.random.random((n_dim)) * vdw_param[0] * 2

	else:
		bond_beads, dxy_index, r_index = update_bond_lists(bond_matrix)

		energy = max_energy + 1

		while energy > max_energy:
			new_vec = ut.rand_vector(n_dim) * vdw_param[0]
			pos[n] = pos[n-1] + new_vec
			dx, dy = get_dx_dy(pos[:bead+1], cell_dim)
			r2 = dx**2 + dy**2

			energy, _ = calc_energy_forces(dx, dy, r2, bond_matrix, vdw_matrix, check_cutoff(r2, rc**2), 
						vdw_param, bond_param, angle_param, rc, bond_beads, dxy_index, r_index)

		#if bead == 3: sys.exit()

	return pos


def create_pos_array(n_dim, n_fibre, l_fibre, vdw_param, bond_param, angle_param, rc):
	"""
	create_pos_array(n_dim, n_fibre, l_fibre, vdw_param, bond_param, angle_param, rc)

	Form initial positional array of beads

	Parameters
	----------

	n_dim:  int
		Number of dimensions in simulation

	n_fibre:  int
		Number of fibres in simulation

	l_fibre:  int
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

	n_bead = n_fibre * l_fibre
	pos = np.zeros((n_bead, n_dim), dtype=float)
	bond_matrix = np.zeros((n_bead, n_bead), dtype=int)
	vdw_matrix = np.zeros(n_bead, dtype=int)

	for bead in range(n_bead):
		if bead % l_fibre == 0: vdw_matrix[bead] += 8
		elif bead % l_fibre == l_fibre-1: vdw_matrix[bead] += 8
		else: vdw_matrix[bead] += 1

	vdw_matrix = np.reshape(np.tile(vdw_matrix, (1, n_bead)), (n_bead, n_bead))

	for bead in range(n_bead): vdw_matrix[bead][bead] = 0

	for fibre in range(n_fibre):
		for bead in range(1, l_fibre):
			n = fibre * l_fibre + bead
			bond_matrix[n][n-1] = 1
			bond_matrix[n-1][n] = 1

	init_pos = np.zeros((l_fibre, n_dim), dtype=float)

	print("Creating fibre template containing {} beads".format(l_fibre))

	for bead in range(l_fibre):
		init_pos = grow_fibre(bead, bead, n_dim, n_bead, init_pos, 
				bond_matrix[[slice(0, bead+1) for _ in bond_matrix.shape]],
				vdw_matrix[[slice(0, bead+1) for _ in vdw_matrix.shape]], 
				vdw_param, bond_param, angle_param, rc, 1E2)

	pos[range(l_fibre)] += init_pos
	pos -= np.min(pos)

	size_x = np.max(pos.T[0]) + vdw_param[0] * 2
	size_y = np.max(pos.T[1]) + vdw_param[0] * 2
	bead_list = np.arange(0, l_fibre)

	for fibre in range(1, n_fibre):
		sys.stdout.write("Teselating {} fibres containing {} beads\r".format(fibre, l_fibre))
		sys.stdout.flush()

		pos_x = pos.T[0][bead_list] + size_x * int(fibre / np.sqrt(n_fibre))
		pos_y = pos.T[1][bead_list] + size_y * int(fibre % np.sqrt(n_fibre))

		pos[bead_list + l_fibre * fibre] += np.array((pos_x, pos_y)).T

	cell_dim = np.array([np.max(pos.T[0]) + vdw_param[0], np.max(pos.T[1]) + vdw_param[0]])

	return pos, cell_dim, bond_matrix, vdw_matrix


def setup(pos, cell_dim, bond_matrix, vdw_matrix, mass, vdw_param, bond_param, angle_param, rc, kBT=1.0):
	"""
	setup(cell_dim, nchain, lchain, mass, kBT, vdw_param, bond_param, angle_param, rc)
	
	Setup simulation using parameters provided

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	mass:  float
		Mass of each bead in collagen simulations

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

	vel: array_like, dtype=float
		Velocity of each bead in all collagen fibres

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
	vel = (np.random.random((n_bead, 2)) - 0.5) * 2 * kBT
	dx, dy = get_dx_dy(pos, cell_dim)
	r2 = dx**2 + dy**2

	verlet_list = check_cutoff(r2, rc**2)

	bond_beads, dxy_index, r_index = update_bond_lists(bond_matrix)
	_, frc = calc_energy_forces(dx, dy, r2, bond_matrix, vdw_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxy_index, r_index)

	return vel, frc, verlet_list, bond_beads, dxy_index, r_index


def velocity_verlet_alg(n_dim, pos, vel, frc, mass, bond_matrix, vdw_matrix, verlet_list, bond_beads, dxy_index, 
					r_index, dt, cell_dim, vdw_param, bond_param, angle_param, rc, kBT=1.0, 
					Langevin=False, gamma=0, sigma=1.0, xi = 0, theta = 0):
	"""
	velocity_verlet_alg(n_dim, pos, vel, frc, mass, bond_matrix, vdw_matrix, verlet_list, bond_beads, dxy_index, 
					r_index, dt, cell_dim, vdw_param, bond_param, angle_param, rc, kBT=1.0, 
					Langevin=False, gamma=0, sigma=1.0, xi = 0, theta = 0)

	Integrate positions and velocities through time using verlocity verlet algorithm

	Parameters
	----------

	n_dim:  int
		Number of dimensions

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Velocity of each bead in all collagen fibres

	frc: array_like, dtype=float
		Forces acting upon each bead in all collagen fibres

	mass:  float
		Mass of each bead in collagen simulations

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance

	bond_beads:  array_like, (int); shape=(n_angle, 3)
		Array containing indicies in pos array all 3-bead angular interactions

	dxy_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in dx and dy arrays of all bonded interactions

	r_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in r array of all bonded interactions

	dt:  float
		Length of time step

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

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

	Langevin:  bool
		Whether Langevin thermostat is employed

	gamma: float
		Friction parameter in Langevin thermostat

	sigma: float
		Thermal parameter in Langevin thermostat

	xi:  float
		First random paramter for Langevin thermostat

	theta:  float
		Second random paramter for Langevin thermostat

	
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Updated positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Updated velocity of each bead in all collagen fibres

	frc: array_like, dtype=float
		Updated forces acting upon each bead in all collagen fibres

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance

	tot_energy:  float
		Total energy of simulation

	"""

	n_bead = pos.shape[0]
	
	vel += 0.5 * dt * frc / mass
	pos += dt * vel

	if Langevin:
		C = 0.5 * dt**2 * (frc / mass - gamma * vel) + sigma * dt**(3./2) * (xi + theta / SQRT3) / 2.
		pos += C
	
	cell = np.tile(cell_dim, (n_bead, 1)) 
	pos += cell * (1 - np.array((pos + cell) / cell, dtype=int))
 
	dx, dy = get_dx_dy(pos, cell_dim)
	r2 = dx**2 + dy**2

	verlet_list = check_cutoff(r2, rc**2)
	pot_energy, new_frc = calc_energy_forces(dx, dy, r2, bond_matrix, vdw_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxy_index, r_index)
	vel += 0.5 * dt * new_frc / mass

	if Langevin: vel += 0.5 * dt * frc / mass - gamma * (dt * vel + C) + sigma * xi * np.sqrt(dt)
	
	frc = new_frc
	tot_energy = pot_energy + kin_energy(vel)

	return pos, vel, frc, verlet_list, tot_energy



