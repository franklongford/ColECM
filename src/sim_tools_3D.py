"""
ColECM: Collagen ExtraCellular Matrix Simulation
SIMULATION 3D ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 12/04/2018
"""

import numpy as np
import sys, os, pickle

import utilities as ut

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)


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
	n_dim = vector.shape[1]

	temp_vector = np.reshape(vector, (int(n_vector/2), 2, n_dim))

	"Calculate |rij||rjk| product for each pair of vectors"
	r_prod = np.prod(np.reshape(r_vector, (int(n_vector/2), 2)), axis = 1)

	"Form dot product of each vector pair rij*rjk in vector array corresponding to an angle"
	dot_prod = np.sum(np.prod(temp_vector, axis=1), axis=1)

	"Form pseudo-cross product of each vector pair rij*rjk in vector array corresponding to an angle"
	temp_vector = np.moveaxis(temp_vector, (1, 0, 2), (0, 1, 2))
	cross_prod = np.cross(temp_vector[0], temp_vector[1])

	"Calculate cos(theta) for each angle"
	cos_the = dot_prod / r_prod

	"Calculate sin(theta) for each angle"
	sin_the = cross_prod / np.reshape(np.repeat(r_prod, n_dim), cross_prod.shape)

	return cos_the, sin_the, r_prod


def calc_energy_forces(distances, r2, param, bond_matrix, vdw_matrix, verlet_list, bond_beads, dist_index, r_index):
	"""
	calc_energy_forces(distances, r2, param, bond_matrix, vdw_matrix, verlet_list, bond_beads, dist_index, r_index)

	Return tot potential energy and forces on each bead in simulation

	Parameters
	----------

	dxy:  array_like (float); shape=(2, n_bead, n_bead)
		Displacement along x and y axis between each bead

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

	n_bead = distances.shape[1]
	#frc_beads = np.zeros((n_dim, n_bead))
	f_beads_x = np.zeros((n_bead))
	f_beads_y = np.zeros((n_bead))
	f_beads_z = np.zeros((n_bead))
	pot_energy = 0
	cut_frc = ut.force_vdw(param['rc']**2, param['vdw_sigma'], param['vdw_epsilon'])
	cut_pot = ut.pot_vdw(param['rc']**2, param['vdw_sigma'], param['vdw_epsilon'])
	virial_tensor = np.zeros((3, 3))
	
	nbond = int(np.sum(np.triu(bond_matrix)))

	if nbond > 0:
		"Bond Lengths"
		bond_index_half = np.argwhere(np.triu(bond_matrix))
		bond_index_full = np.argwhere(bond_matrix)

		indices_half = ut.create_index(bond_index_half)
		indices_full = ut.create_index(bond_index_full)
	
		indices_dxy = ut.create_index(dist_index)

		r_half = np.sqrt(r2[indices_half])
		r_full = np.repeat(r_half, 2)
		
		bond_pot = ut.pot_harmonic(r_half, param['bond_r0'], param['bond_k'])
		pot_energy += np.sum(bond_pot)

		bond_frc = ut.force_harmonic(r_half, param['bond_r0'], param['bond_k'])
		for i, sign in enumerate([1, -1]):
			f_beads_x[indices_half[i]] += sign * (bond_frc * distances[0][indices_half] / r_half)
			f_beads_y[indices_half[i]] += sign * (bond_frc * distances[1][indices_half] / r_half)
			f_beads_z[indices_half[i]] += sign * (bond_frc * distances[2][indices_half] / r_half)

		#for i in range(3):
		#	for j in range(3): virial_tensor[i][j] += np.sum(bond_frc / r_half * distances[i][indices_half] * distances[j][indices_half])

		"Bond Angles"
		try:
			"Make array of vectors rij, rjk for all connected bonds"
			vector = np.stack((distances[0][indices_dxy], distances[1][indices_dxy], distances[2][indices_dxy]), axis=1)
			n_vector = vector.shape[0]

			"Find |rij| values for each vector"
			r_vector = r_half[r_index]
			cos_the, sin_the, r_prod = cos_sin_theta(vector, r_vector)
			pot_energy += param['angle_k'] * np.sum(cos_the + 1)

			"Form arrays of |rij| vales, cos(theta) and |rij||rjk| terms same shape as vector array"
			r_array = np.reshape(np.repeat(r_vector, 3), vector.shape)
			sin_the_array = np.reshape(np.repeat(sin_the, 2), vector.shape)
			r_prod_array = np.reshape(np.repeat(r_prod, 6), vector.shape)

			"Form left and right hand side terms of (cos(theta) rij / |rij|^2 - rjk / |rij||rjk|)"
			r_left = vector / r_prod_array
			r_right = sin_the_array * vector / r_array**2

			ij_indices = np.arange(0, n_vector, 2)
			jk_indices = np.arange(1, n_vector, 2)

			"Perfrom right hand - left hand term for every rij rkj pair"
			r_left[ij_indices] -= r_right[jk_indices]
			r_left[jk_indices] -= r_right[ij_indices] 
		
			"Calculate forces upon beads i, j and k"
			frc_angle_ij = param['angle_k'] * r_left
			frc_angle_k = -np.sum(np.reshape(frc_angle_ij, (int(n_vector/2), 2, 3)), axis=1)

			"Add angular forces to force array" 
			f_beads_x[bond_beads.T[0]] += frc_angle_ij[ij_indices].T[0]
			f_beads_x[bond_beads.T[1]] += frc_angle_k.T[0]
			f_beads_x[bond_beads.T[2]] += frc_angle_ij[jk_indices].T[0]

			f_beads_y[bond_beads.T[0]] += frc_angle_ij[ij_indices].T[1]
			f_beads_y[bond_beads.T[1]] += frc_angle_k.T[1]
			f_beads_y[bond_beads.T[2]] += frc_angle_ij[jk_indices].T[1]

			f_beads_z[bond_beads.T[0]] += frc_angle_ij[ij_indices].T[2]
			f_beads_z[bond_beads.T[1]] += frc_angle_k.T[2]
			f_beads_z[bond_beads.T[2]] += frc_angle_ij[jk_indices].T[2]

		except IndexError: pass

	non_zero = np.nonzero(r2 * verlet_list)
	nonbond_pot = vdw_matrix[non_zero] * ut.pot_vdw((r2 * verlet_list)[non_zero], param['vdw_sigma'], param['vdw_epsilon']) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	nonbond_frc = vdw_matrix[non_zero] * ut.force_vdw((r2 * verlet_list)[non_zero], param['vdw_sigma'], param['vdw_epsilon']) - cut_frc
	temp_xyz = np.zeros(distances.shape)
	
	for i in range(3):
		temp_xyz[i][non_zero] += nonbond_frc * (distances[i][non_zero] / r2[non_zero])
		for j in range(3): virial_tensor[i][j] += np.sum(np.triu(temp_xyz[i] * distances[i] * distances[j]))

	f_beads_x += np.sum(temp_xyz[0], axis=0)
	f_beads_y += np.sum(temp_xyz[1], axis=0)
	f_beads_z += np.sum(temp_xyz[2], axis=0)

	frc_beads = np.transpose(np.array([f_beads_x, f_beads_y, f_beads_z]))

	return pot_energy, frc_beads, virial_tensor


def velocity_verlet_alg(pos, vel, frc, virial_tensor, param, bond_matrix, vdw_matrix, verlet_list, bond_beads, dist_index, 
			r_index, dt, sqrt_dt, cell_dim, NPT=False):
	"""
	velocity_verlet_alg(pos, vel, frc, virial_tensor, param, bond_matrix, vdw_matrix, verlet_list, bond_beads, dist_index, 
			r_index, dt, sqrt_dt, cell_dim, NPT=False)

	Integrate positions and velocities through time using verlocity verlet algorithm

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Velocity of each bead in all collagen fibres

	frc: array_like, dtype=float
		Forces acting upon each bead in all collagen fibres

	virial_tensor: 

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance

	bond_beads:  array_like, (int); shape=(n_angle, 3)
		Array containing indicies in pos array all 3-bead angular interactions

	dist_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in dx and dy arrays of all bonded interactions

	r_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in r array of all bonded interactions

	dt:  float
		Length of time step

	sqrt_dt:  float
		Sqare root of time step

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	vdw_param: array_like, dtype=float
		Parameters of van der Waals potential (sigma, epsilon)

	NPT:  

	
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

	pot_energy:  float
		Total energy of simulation

	virial_tensor: 

	"""

	beta = np.random.normal(0, 1, (param['n_bead'], param['n_dim']))
	vel += frc / param['mass'] * dt

	if NPT:
		kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
		P_t = 1 / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
		mu = (1 + (param['lambda_p'] * (P_t - param['P_0'])))**(1./3) 

	d_vel = param['sigma'] * beta - param['gamma'] * vel
	pos += (vel + 0.5 * d_vel) * dt
	vel += d_vel

	if NPT:
		pos = mu * pos
		cell_dim = mu * cell_dim

	cell = np.tile(cell_dim, (param['n_bead'], 1)) 
	pos += cell * (1 - np.array((pos + cell) / cell, dtype=int))

	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)

	verlet_list = ut.check_cutoff(r2, param['rc']**2)
	pot_energy, frc, virial_tensor = calc_energy_forces(distances, r2, param, bond_matrix, vdw_matrix, verlet_list, bond_beads, dist_index, r_index)

	return pos, vel, frc, cell_dim, verlet_list, pot_energy, virial_tensor
