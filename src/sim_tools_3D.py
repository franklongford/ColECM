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


def calc_energy_forces(pos, cell_dim, bond_indices, angle_indices, angle_bond_indices, param):
	"""
	calc_energy_forces(pos, cell_dim, bond_indices, angle_indices, angle_bond_indices, vdw_coeff, param)

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

	virial_tensor:  array_like (float); shape=(n_dim, n_dim)
		Virial term of pressure tensor components

	"""

	f_beads = np.zeros((3, pos.shape[0]))
	pot_energy = 0
	cut_frc = ut.force_vdw(param['rc']**2, param['vdw_sigma'], param['vdw_epsilon'])
	cut_pot = ut.pot_vdw(param['rc']**2, param['vdw_sigma'], param['vdw_epsilon'])
	virial_tensor = np.zeros((3, 3))
	n_bond = bond_indices[0].shape[0]

	pair_dist = ut.get_distances(pos, cell_dim)
	pair_r2 = np.sum(pair_dist**2, axis=0)

	if n_bond > 0:

		"Bond Lengths"
		bond_r = np.sqrt(pair_r2[bond_indices])
		#verlet_list_r0 = ut.check_cutoff(r_half, param['bond_r0'])
		#verlet_list_r1 = ut.check_cutoff(r_half, param['bond_r1'])

		bond_pot = ut.pot_harmonic(bond_r, param['bond_r0'], param['bond_matrix'][bond_indices])# * verlet_list_r0
		#bond_pot_1 = ut.pot_harmonic(r_half, param['bond_r1'], param['bond_k1']) * verlet_list_r1
		pot_energy += 0.5 * np.sum(bond_pot)# + np.sum(bond_pot_1)

		bond_frc = ut.force_harmonic(bond_r, param['bond_r0'], param['bond_matrix'][bond_indices])# * verlet_list_r0
		#bond_frc_1 = ut.force_harmonic(r_half, param['bond_r1'], param['bond_k1']) * verlet_list_r1

		temp_frc = np.zeros((3, pos.shape[0], pos.shape[0]))
		for i in range(3): 
			temp_frc[i][bond_indices] += bond_frc * pair_dist[i][bond_indices] / bond_r
			f_beads[i] += np.sum(temp_frc[i], axis=1)

		#for i in range(3):
		#	for j in range(3): virial_tensor[i][j] += np.sum(bond_frc / r_half * distances[i][indices_half] * distances[j][indices_half])

		"Bond Angles"
		try:
			angle_dist = pair_dist.T[angle_bond_indices].T

			"Make array of vectors rij, rjk for all connected bonds"
			vector = np.stack((angle_dist[0], angle_dist[1], angle_dist[2]), axis=1)
			n_vector = int(vector.shape[0])

			"Find |rij| values for each vector"
			r_vector = np.sqrt(pair_r2[angle_bond_indices])
			cos_the, sin_the, r_prod = cos_sin_theta(vector, r_vector)
			pot_energy += np.sum(param['angle_array'] * (cos_the + 1))

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
			frc_angle_ij = param['angle_k0'] * r_left
			frc_angle_k = -np.sum(np.reshape(frc_angle_ij, (int(n_vector/2), 2, 3)), axis=1)

			"Add angular forces to force array" 
			for i in range(3): 
				f_beads[i][angle_indices.T[0]] -= frc_angle_ij[ij_indices].T[i]
				f_beads[i][angle_indices.T[1]] -= frc_angle_k.T[i]
				f_beads[i][angle_indices.T[2]] -= frc_angle_ij[jk_indices].T[i]

		except IndexError: pass
	
	verlet_list = ut.check_cutoff(pair_r2, param['rc']**2)
	non_zero = np.nonzero(pair_r2 * verlet_list)

	nonbond_pot = ut.pot_vdw((pair_r2 * verlet_list)[non_zero], param['vdw_sigma'], param['vdw_matrix'][non_zero]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	nonbond_frc = ut.force_vdw((pair_r2 * verlet_list)[non_zero], param['vdw_sigma'], param['vdw_matrix'][non_zero]) - cut_frc
	temp_xyz = np.zeros(pair_dist.shape)
	
	for i in range(3):
		temp_xyz[i][non_zero] += nonbond_frc * (pair_dist[i][non_zero] / pair_r2[non_zero])
		for j in range(3):
			virial_tensor[i][j] += np.sum(np.triu(temp_xyz[i] * pair_dist[i] * pair_dist[j]))

		f_beads[i] += np.sum(temp_xyz[i], axis=0)

	frc = f_beads.T
	
	return frc, pot_energy, virial_tensor


def calc_energy_forces_mpi(pos, cell_dim, pos_indices, bond_indices, glob_indices, angle_indices, angle_bond_indices, 
				angle_coeff, vdw_coeff, virial_indicies, param):
	"""
	calc_energy_forces(distances, r2, bond_matrix, vdw_matrix, verlet_list, bond_beads, dist_index, r_index, param)

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

	virial_tensor:  array_like (float); shape=(n_dim, n_dim)
		Virial term of pressure tensor components

	"""

	f_beads = np.zeros((3, pos.shape[0]))
	pot_energy = 0
	cut_frc = ut.force_vdw(param['rc']**2, param['vdw_sigma'], param['vdw_epsilon'])
	cut_pot = ut.pot_vdw(param['rc']**2, param['vdw_sigma'], param['vdw_epsilon'])
	virial_tensor = np.zeros((3, 3))
	n_bond = bond_indices[0].shape[0]

	pair_dist = ut.get_distances_mpi(pos, pos_indices, cell_dim)
	pair_r2 = np.sum(pair_dist**2, axis=0)

	if n_bond > 0:

		"Bond Lengths"
		bond_r = np.sqrt(pair_r2[bond_indices])
		#verlet_list_r0 = ut.check_cutoff(r_half, param['bond_r0'])
		#verlet_list_r1 = ut.check_cutoff(r_half, param['bond_r1'])

		bond_pot = ut.pot_harmonic(bond_r, param['bond_r0'], param['bond_matrix'][glob_indices])# * verlet_list_r0
		#bond_pot_1 = ut.pot_harmonic(r_half, param['bond_r1'], param['bond_k1']) * verlet_list_r1
		pot_energy += 0.5 * np.sum(bond_pot)# + np.sum(bond_pot_1)

		bond_frc = ut.force_harmonic(bond_r, param['bond_r0'], param['bond_matrix'][glob_indices])# * verlet_list_r0
		#bond_frc_1 = ut.force_harmonic(r_half, param['bond_r1'], param['bond_k1']) * verlet_list_r1

		temp_frc = np.zeros((3, pos.shape[0], pos.shape[0]))
		for i in range(3): 
			temp_frc[i][glob_indices] += bond_frc * pair_dist[i][bond_indices] / bond_r
			f_beads[i] += np.sum(temp_frc[i], axis=1)

		#for i in range(3):
		#	for j in range(3): virial_tensor[i][j] += np.sum(bond_frc / r_half * distances[i][indices_half] * distances[j][indices_half])

		"Bond Angles"
		try:
			angle_dist = (pos[angle_bond_indices[1]] - pos[angle_bond_indices[0]]).T
			for i in range(param['n_dim']): angle_dist[i] -= cell_dim[i] * np.array(2 * angle_dist[i] / cell_dim[i], dtype=int)
			angle_r2 = np.sum(angle_dist**2, axis=0)

			"Make array of vectors rij, rjk for all connected bonds"
			vector = np.stack((angle_dist[0], angle_dist[1], angle_dist[2]), axis=1)
			n_vector = int(vector.shape[0])

			"Find |rij| values for each vector"
			r_vector = np.sqrt(angle_r2)
			cos_the, sin_the, r_prod = cos_sin_theta(vector, r_vector)
			pot_energy += np.sum(angle_coeff * (cos_the + 1))

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
			frc_angle_ij = param['angle_k0'] * r_left
			frc_angle_k = -np.sum(np.reshape(frc_angle_ij, (int(n_vector/2), 2, 3)), axis=1)

			"Add angular forces to force array" 
			for i in range(3): 
				f_beads[i][angle_indices.T[0]] -= frc_angle_ij[ij_indices].T[i]
				f_beads[i][angle_indices.T[1]] -= frc_angle_k.T[i]
				f_beads[i][angle_indices.T[2]] -= frc_angle_ij[jk_indices].T[i]

		except IndexError: pass
	
	verlet_list = ut.check_cutoff(pair_r2, param['rc']**2)
	non_zero = np.nonzero(pair_r2 * verlet_list)

	nonbond_pot = ut.pot_vdw((pair_r2 * verlet_list)[non_zero], param['vdw_sigma'], vdw_coeff[non_zero]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	nonbond_frc = ut.force_vdw((pair_r2 * verlet_list)[non_zero], param['vdw_sigma'], vdw_coeff[non_zero]) - cut_frc
	temp_xyz = np.zeros(pair_dist.shape)
	
	for i in range(3):
		temp_xyz[i][non_zero] += nonbond_frc * (pair_dist[i][non_zero] / pair_r2[non_zero])
		for j in range(3):
			virial_tensor[i][j] += np.sum(np.triu(temp_xyz[i] * pair_dist[i] * pair_dist[j])[virial_indicies])

		f_beads[i] += np.sum(temp_xyz[i], axis=0)

	frc = f_beads.T
	
	return frc, pot_energy, virial_tensor
