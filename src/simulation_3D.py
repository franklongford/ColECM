"""
COLLAGEN FIBRE SIMULATION 3D

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 09/03/2018
"""

import numpy as np
import random

import sys, os, pickle

import utilities as ut
from setup import grow_fibre

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


def calc_energy_forces(dxdydz, r2, bond_matrix, vdw_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxdy_index, r_index):
	"""
	calc_energy_forces(dxdy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxdy_index, r_index)

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

	n_bead = dxdydz.shape[1]
	#frc_beads = np.zeros((n_dim, n_bead))
	f_beads_x = np.zeros((n_bead))
	f_beads_y = np.zeros((n_bead))
	f_beads_z = np.zeros((n_bead))
	pot_energy = 0
	cut_frc = ut.force_vdw(rc**2, vdw_param[0], vdw_param[1])
	cut_pot = ut.pot_vdw(rc**2, vdw_param[0], vdw_param[1])
	
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
		
		bond_pot = ut.pot_harmonic(r_half, bond_param[0], bond_param[1])
		pot_energy += np.sum(bond_pot)

		bond_frc = ut.force_harmonic(r_half, bond_param[0], bond_param[1])
		for i, sign in enumerate([1, -1]):
			f_beads_x[indices_half[i]] += sign * (bond_frc * dxdydz[0][indices_half] / r_half)
			f_beads_y[indices_half[i]] += sign * (bond_frc * dxdydz[1][indices_half] / r_half)
			f_beads_z[indices_half[i]] += sign * (bond_frc * dxdydz[2][indices_half] / r_half)

		"Bond Angles"

		try:
			"Make array of vectors rij, rjk for all connected bonds"
			vector = np.stack((dxdydz[0][indices_dxy], dxdydz[1][indices_dxy], dxdydz[2][indices_dxy]), axis=1)
			n_vector = vector.shape[0]

			"Find |rij| values for each vector"
			r_vector = r_half[r_index]
			cos_the, sin_the, r_prod = cos_sin_theta(vector, r_vector)
			pot_energy += angle_param[1] * np.sum(cos_the + 1)

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
			frc_angle_ij = angle_param[1] * r_left
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
	nonbond_pot = vdw_matrix[non_zero] * ut.pot_vdw((r2 * verlet_list)[non_zero], vdw_param[0], vdw_param[1]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	nonbond_frc = vdw_matrix[non_zero] * ut.force_vdw((r2 * verlet_list)[non_zero], vdw_param[0], vdw_param[1]) - cut_frc
	temp_x = np.zeros(r2.shape)
	temp_y = np.zeros(r2.shape)
	temp_z = np.zeros(r2.shape)

	temp_x[non_zero] += nonbond_frc * (dxdydz[0][non_zero] / r2[non_zero])
	temp_y[non_zero] += nonbond_frc * (dxdydz[1][non_zero] / r2[non_zero])
	temp_z[non_zero] += nonbond_frc * (dxdydz[2][non_zero] / r2[non_zero])

	f_beads_x += np.sum(temp_x, axis=0)
	f_beads_y += np.sum(temp_y, axis=0)
	f_beads_z += np.sum(temp_z, axis=0)

	frc_beads = np.transpose(np.array([f_beads_x, f_beads_y, f_beads_z]))

	return pot_energy, frc_beads


def create_pos_array(n_dim, n_fibril_x, n_fibril_y, l_fibril, vdw_param, bond_param, angle_param, rc):
	"""
	create_pos_array(n_dim, n_fibril_x, n_fibril_y, l_fibril, vdw_param, bond_param, angle_param, rc)

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

	if ('-nfibrilz' in sys.argv): n_fibril_z = int(sys.argv[sys.argv.index('-nfibrilz') + 1])
	else: n_fibril_z = int(input("Enter number of fibrils in z dimension: "))

	n_fibril = n_fibril_x * n_fibril_y * n_fibril_z
	n_bead = n_fibril * l_fibril
	pos = np.zeros((n_bead, n_dim), dtype=float)
	bond_matrix = np.zeros((n_bead, n_bead), dtype=int)
	vdw_matrix = np.zeros(n_bead, dtype=int)

	for bead in range(n_bead):
		if bead % l_fibril == 0: vdw_matrix[bead] += 8
		elif bead % l_fibril == l_fibril-1: vdw_matrix[bead] += 8
		else: vdw_matrix[bead] += 1

	vdw_matrix = np.reshape(np.tile(vdw_matrix, (1, n_bead)), (n_bead, n_bead))

	for bead in range(n_bead): vdw_matrix[bead][bead] = 0

	for fibril in range(n_fibril):
		for bead in range(1, l_fibril):
			n = fibril * l_fibril + bead
			bond_matrix[n][n-1] = 1
			bond_matrix[n-1][n] = 1

	init_pos = np.zeros((l_fibril, n_dim), dtype=float)

	print("Creating fibril template containing {} beads".format(l_fibril))

	for bead in range(l_fibril):
		init_pos = grow_fibre(bead, bead, n_dim, n_bead, init_pos, 
				bond_matrix[[slice(0, bead+1) for _ in bond_matrix.shape]],
				vdw_matrix[[slice(0, bead+1) for _ in vdw_matrix.shape]], 
				vdw_param, bond_param, angle_param, rc, 1E2)

	pos[range(l_fibril)] += init_pos
	pos -= np.min(pos)

	size_x = np.max(pos.T[0]) + vdw_param[0] 
	size_y = np.max(pos.T[1]) + vdw_param[0] 
	size_z = np.max(pos.T[2]) + vdw_param[0] 
	bead_list = np.arange(0, l_fibril)

	for k in range(n_fibril_z):
		for i in range(n_fibril_x):
			for j in range(n_fibril_y):
				if k + j + i == 0: continue
				sys.stdout.write("Teselating {} fibres containing {} beads\r".format(fibril, l_fibril))
				sys.stdout.flush()

				fibril = (j + i * n_fibril_y + k * n_fibril_x * n_fibril_y)

				pos_x = pos.T[0][bead_list] + size_x * i
				pos_y = pos.T[1][bead_list] + size_y * j
				pos_z = pos.T[2][bead_list] + size_z * k

				pos[bead_list + l_fibril * fibril] += np.array((pos_x, pos_y, pos_z)).T

	cell_dim = np.array([np.max(pos.T[0]) + vdw_param[0], np.max(pos.T[1]) + vdw_param[0], np.max(pos.T[2]) + vdw_param[0]])

	return pos, cell_dim, bond_matrix, vdw_matrix


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
 
	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)

	verlet_list = ut.check_cutoff(r2, rc**2)
	pot_energy, new_frc = calc_energy_forces(distances, r2, bond_matrix, vdw_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxy_index, r_index)
	vel += 0.5 * dt * new_frc / mass

	if Langevin: vel += 0.5 * dt * frc / mass - gamma * (dt * vel + C) + sigma * xi * np.sqrt(dt)
	
	frc = new_frc
	tot_energy = pot_energy + ut.kin_energy(vel, mass, n_dim)

	return pos, vel, frc, verlet_list, tot_energy



