"""
COLLAGEN FIBRE SIMULATION 2D

Created by: Frank Longford
Created on: 01/11/15

Last Modified: 06/03/2018
"""

import numpy as np
import random

import sys
import os



SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)


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


def check_cutoff(array, thresh):
	"""
	check_cutoff(array, rc)

	Determines whether elements of array are less than or equal to thresh
	"""

	return (array <= thresh).astype(float)


def get_dx_dy(pos, N, cell_dim):
	"""
	get_dx_dy(pos, N, cell_dim)

	Calculate distance vector between two beads
	"""

	N = pos.shape[0]
	temp_pos = np.moveaxis(pos, 0, 1)

	dx = np.tile(temp_pos[0], (N, 1))
	dy = np.tile(temp_pos[1], (N, 1))

	dx = dx.T - dx
	dy = dy.T - dy

	dx -= cell_dim[0] * np.array(2 * dx / cell_dim[0], dtype=int)
	dy -= cell_dim[1] * np.array(2 * dy / cell_dim[1], dtype=int)

	return dx, dy


def create_index(array):
	"""
	create_index(array)

	Takes a list of 2D indicies and returns an index array that can be used to access elements in a 2D numpy array
	"""
    
	return (np.array(array.T[0]), np.array(array.T[1]))


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


def kin_energy(vel): return np.mean(vel**2)


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

	atoms = []
	dxdy_index = []
	r_index = []

	count = np.unique(bond_index_full.T[0]).shape[0]

	for n in range(N):
		slice_full = np.argwhere(bond_index_full.T[0] == n)
		slice_half = np.argwhere(bond_index_half.T[0] == n)

		if slice_full.shape[0] > 1:
			atoms.append(np.unique(bond_index_full[slice_full].flatten()))
			dxdy_index.append(bond_index_full[slice_full])
			r_index.append(slice_full[0])

	return atoms, dxdy_index, r_index


def calc_energy_forces_linear(dx, dy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, atoms, dxdy_index, r_index):
	"""
	calc_energy_forces_linear(dx, dy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc)

	Return tot potential energy and forces on each bead in simulation
	"""

	N = dx.shape[0]
	f_beads_x = np.zeros((N))
	f_beads_y = np.zeros((N))
	pot_energy = 0
	cut_frc = force_vdw(rc**2, vdw_param[0], vdw_param[1])
	cut_pot = pot_vdw(rc**2, vdw_param[0], vdw_param[1])
	
	nbond = int(np.sum(np.triu(bond_matrix)))

	if nbond > 0:
		"Bond Lengths"
		bond_index_half = np.argwhere(np.triu(bond_matrix))
		bond_index_full = np.argwhere(bond_matrix)

		indices_half = create_index(bond_index_half)
		indices_full = create_index(bond_index_full)

		r = np.sqrt(r2[indices_half])
		
		bond_pot = pot_harmonic(r, bond_param[0], bond_param[1])
		pot_energy += np.sum(bond_pot)

		bond_frc = force_harmonic(r, bond_param[0], bond_param[1])
		for i, sign in enumerate([1, -1]):
			f_beads_x[indices_half[i]] += sign * (bond_frc * dx[indices_half] / r)
			f_beads_y[indices_half[i]] += sign * (bond_frc * dy[indices_half] / r)

		#"""
		"Bond Angles"
		r = np.repeat(r, 2)
		switch = np.ones((2, 2)) - np.identity(2)

		for i, atom in enumerate(atoms):

			di = create_index(dxdy_index[i])
			ri = r_index[i]
		
			vector = np.concatenate((dx[di], dy[di]), axis=0).T
		
			dot_prod = np.dot(vector[0], vector[1])
			cos_the = dot_prod / r[ri]

			pot_energy += np.nansum(np.triu(cos_the) * angle_param[1] * switch)

			vector_norm = vector / r[ri]
				
			frc_angle = angle_param[1] * (vector_norm * cos_the - np.flip(vector_norm, axis=0)) / r[ri]
			frc_angle = np.insert(frc_angle, -1, -np.sum(frc_angle, axis=0), axis=0)

			f_beads_x[atom] -= frc_angle.T[0]
			f_beads_y[atom] -= frc_angle.T[1]

		#"""

	non_zero = np.nonzero(r2)
	nonbond_pot = pot_vdw(r2[non_zero] * verlet_list[non_zero], vdw_param[0], vdw_param[1]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	nonbond_frc = force_vdw(r2 * verlet_list, vdw_param[0], vdw_param[1]) - cut_frc
	f_beads_x += np.nansum(nonbond_frc * dx / r2, axis=0)
	f_beads_y += np.nansum(nonbond_frc * dy / r2, axis=0)

	frc_beads = np.transpose(np.array([f_beads_x, f_beads_y]))

	return pot_energy, frc_beads


def grow_fibre(n, bead, N, pos, bond_matrix, vdw_param, bond_param, angle_param, rc, cell_dim, n_section, limits, max_energy):
	"""
	grow_fibre(n, bead, N, pos, bond_matrix, vdw_param, bond_param, angle_param, rc, cell_dim, n_section, limits, max_energy)

	Grow collagen fibre consisting of beads
	"""

	if bead == 0:
		pos[n] = (np.random.random((2)) * cell_dim / n_section + limits)

	else:
		bond_matrix[n][n-1] = 1
		bond_matrix[n-1][n] = 1

		atoms, dxdy_index, r_index = update_bond_lists(bond_matrix)

		energy = max_energy + 1

		while energy > max_energy:
			print(energy)
			new_vec = rand_vector(2) * vdw_param[0]
			pos[n] = pos[n-1] + new_vec
			dx, dy = get_dx_dy(np.array(pos), N, cell_dim)
			r2 = dx**2 + dy**2

			energy, _ = calc_energy_forces_linear(dx, dy, r2, bond_matrix, check_cutoff(r2, rc**2), vdw_param, bond_param, angle_param, rc, atoms, dxdy_index, r_index)
			
	return pos, bond_matrix


def setup(cell_dim, n_fibre, l_fibre, mass, kBT, vdw_param, bond_param, angle_param, rc):
	"""
	setup(cell_dim, nchain, lchain, mass, kBT, vdw_param, bond_param, angle_param, rc)
	
	Setup simulation using parameters provided

	Parameters
	----------

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	n_fibre: int
		Number of collegen fibres to populate

	l_fibre: int
		Length of each fibre in number of beads

	mass: float
		Mass of each bead in collagen simulations

	kBT: float
		Value of thermostat constant kB x T in reduced units

	vdw_param: array_like, dtype=float
		Parameters of van der Waals potential (sigma, epsilon)

	bond_param: array_like, dtype=float
		Parameters of bond length potential (r0, kB)

	angle_param: array_like, dtype=float
		Parameters of angular potential (theta0, kA)

	rc:  float
		Radial cutoff distance for non-bonded interactions

	
	Returns
	-------

	pos: array_like, dtype=float
		Positions of each bead in all collagen fibres

	vel: array_like, dtype=float
		Velocity of each bead in all collagen fibres

	frc: array_like, dtype=float
		Forces acting upon each bead in all collagen fibres

	bond_matrix: array_like, dtype=int
		Matrix determining whether a bond is present between two beads

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance
	
	"""

	N = n_fibre * l_fibre
	pos = np.zeros((N, 2), dtype=float)
	bond_matrix = np.zeros((N, N), dtype=int)

	if n_fibre > 1: n_section = np.sqrt(np.min([i for i in np.arange(n_fibre + 1)**2 if i >= n_fibre]))
	else: n_section = 1

	sections = np.arange(n_section**2)

	for fibre in range(n_fibre):
		section = random.choice(sections)
		sections = remove_element(section, sections)

		lim_x = cell_dim[0] / n_section * (section % n_section)
		lim_y = cell_dim[1] / n_section * int(section / n_section)
		limits = np.array([lim_x, lim_y])

		for bead in range(l_fibre):
			n = fibre * l_fibre + bead
			pos, bond_matrix = grow_fibre(n, bead, N, pos, bond_matrix, vdw_param, bond_param, 
						      angle_param, rc, cell_dim, n_section, limits, 1E2)

	vel = (np.random.random((N, 2)) - 0.5) * 2 * kBT
	dx, dy = get_dx_dy(pos, N, cell_dim)
	r2 = dx**2 + dy**2

	verlet_list = check_cutoff(r2, rc**2)

	atoms, dxdy_index, r_index = update_bond_lists(bond_matrix)
	_, frc = calc_energy_forces_linear(dx, dy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, atoms, dxdy_index, r_index)

	return pos, vel, frc, bond_matrix, verlet_list, atoms, dxdy_index, r_index


def grow_fibre_test(n, bead, N, pos, bond_matrix, vdw_param, bond_param, angle_param, rc, cell_dim, max_energy):
	"""
	grow_fibre(n, bead, N, pos, bond_matrix, vdw_param, bond_param, angle_param, rc, cell_dim, max_energy)

	Grow collagen fibre consisting of beads
	"""

	if bead == 0:
		pos[n] = (np.random.random((2)) * cell_dim / 5.)

	else:
		bond_matrix[n][n-1] = 1
		bond_matrix[n-1][n] = 1

		atoms, dxdy_index, r_index = update_bond_lists(bond_matrix)

		energy = max_energy + 1

		while energy > max_energy:
			print(energy)
			new_vec = rand_vector(2) * vdw_param[0]
			pos[n] = pos[n-1] + new_vec
			dx, dy = get_dx_dy(np.array(pos), N, cell_dim)
			r2 = dx**2 + dy**2

			energy, _ = calc_energy_forces_linear(dx, dy, r2, bond_matrix, check_cutoff(r2, rc**2), vdw_param, bond_param, angle_param, rc, atoms, dxdy_index, r_index)
			
	return pos, bond_matrix


def setup_test(cell_dim, n_fibre, l_fibre, mass, kBT, vdw_param, bond_param, angle_param, rc):
	"""
	setup(cell_dim, nchain, lchain, mass, kBT, vdw_param, bond_param, angle_param, rc)
	
	Setup simulation using parameters provided

	Parameters
	----------

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	n_fibre: int
		Number of collegen fibres to populate

	l_fibre: int
		Length of each fibre in number of beads

	mass: float
		Mass of each bead in collagen simulations

	kBT: float
		Value of thermostat constant kB x T in reduced units

	vdw_param: array_like, dtype=float
		Parameters of van der Waals potential (sigma, epsilon)

	bond_param: array_like, dtype=float
		Parameters of bond length potential (r0, kB)

	angle_param: array_like, dtype=float
		Parameters of angular potential (theta0, kA)

	rc:  float
		Radial cutoff distance for non-bonded interactions

	
	Returns
	-------

	pos: array_like, dtype=float
		Positions of each bead in all collagen fibres

	vel: array_like, dtype=float
		Velocity of each bead in all collagen fibres

	frc: array_like, dtype=float
		Forces acting upon each bead in all collagen fibres

	bond_matrix: array_like, dtype=int
		Matrix determining whether a bond is present between two beads

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance
	
	"""

	N = n_fibre * l_fibre
	pos = np.zeros((N, 2), dtype=float)
	bond_matrix = np.zeros((N, N), dtype=int)

	for bead in range(l_fibre):
		pos, bond_matrix = grow_fibre_test(bead, bead, N, pos, bond_matrix, vdw_param, bond_param, 
					      angle_param, rc, cell_dim, 1E2)

	size_x = np.max(pos.T[0]) + vdw_param[0]
	size_y = np.max(pos.T[1]) + vdw_param[0]
	bead_list = np.arange(0, l_fibre)

	if n_fibre > 1: n_section = np.sqrt(np.min([i for i in np.arange(n_fibre + 1)**2 if i >= n_fibre]))
	else: n_section = 1

	sections = np.arange(n_section**2)


	for fibre in range(1, n_fibre):
	
		pos_x = pos.T[0][bead_list] + size_x * fibre
		pos_y = pos.T[1][bead_list] + size_y * fibre

		pos[bead_list + l_fibre * fibre] += np.array((pos_x, pos_y)).T

		for bead in range(1, l_fibre):
			n = fibre * l_fibre + bead
			bond_matrix[n][n-1] = 1
			bond_matrix[n-1][n] = 1

	cell_dim = np.array([np.max(pos.T[0]) + vdw_param[0], np.max(pos.T[1]) + vdw_param[0]])

	vel = (np.random.random((N, 2)) - 0.5) * 2 * kBT
	dx, dy = get_dx_dy(pos, N, cell_dim)
	r2 = dx**2 + dy**2

	verlet_list = check_cutoff(r2, rc**2)

	atoms, dxdy_index, r_index = update_bond_lists(bond_matrix)
	_, frc = calc_energy_forces_linear(dx, dy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, atoms, dxdy_index, r_index)

	return pos, vel, frc, cell_dim, bond_matrix, verlet_list, atoms, dxdy_index, r_index


def velocity_verlet_alg(pos, vel, frc, mass, bond_matrix, verlet_list, atoms, dxdy_index, r_index, dt, cell_dim, vdw_param, bond_param, angle_param, rc, kBT, gamma, sigma, xi = 0, theta = 0, Langevin=False):
	"""
	velocity_verlet_alg(pos, vel, frc, mass, bond_matrix, verlet_list, dt, cell_dim, vdw_param, bond_param, angle_param, rc, kBT, gamma, sigma, xi = 0, theta = 0, Langevin=False)

	Integrate positions and velocities through time using verlocity verlet algorithm
	"""

	N = pos.shape[0]
	
	vel += 0.5 * dt * frc / mass
	pos += dt * vel

	if Langevin:
		C = 0.5 * dt**2 * (frc / mass - gamma * vel) + sigma * dt**(3./2) * (xi + theta / SQRT3) / 2.
		pos += C
	
	cell = np.tile(cell_dim, (N, 1)) 
	pos += cell * (1 - np.array((pos + cell) / cell, dtype=int))
 
	dx, dy = get_dx_dy(pos, N, cell_dim)
	r2 = dx**2 + dy**2

	verlet_list = check_cutoff(r2, rc**2)
	pot_energy, new_frc = calc_energy_forces_linear(dx, dy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, atoms, dxdy_index, r_index)
	vel += 0.5 * dt * new_frc / mass

	if Langevin: vel += 0.5 * dt * frc / mass - gamma * (dt * vel + C) + sigma * xi * np.sqrt(dt)
	
	frc = new_frc
	tot_energy = pot_energy + kin_energy(vel)

	return pos, vel, frc, verlet_list, tot_energy



def save_traj(pos, vel):

	pass


""" Visualisation of System """

def plot_system(pos, vel, frc, N, L, bsize, n):

	width = 0.2
	positions = np.rot90(np.array(pos))
	fig = plt.figure(0, figsize=(15,15))
	plt.title(n)
	fig.clf()

	#ax = plt.subplot(2,1,1)
	plt.scatter(positions[0], positions[1], c=range(N), s=bsize)
	plt.axis([0, L, 0, L])
	plt.show(False)
	plt.draw()

	"""

	velocities = np.rot90(vel)
	forces = np.rot90(frc)

	fig = plt.figure(1, figsize=(15,15))
	#fig.clf()

	ax = plt.subplot(2,1,1)
	ax.set_ylim(-10,10)
	vel_x = ax.bar(range(N), velocities[0], width, color='b')
	vel_y = ax.bar(np.arange(N)+width, velocities[1], width, color='g')

	ax = plt.subplot(2,1,2)
	ax.set_ylim(-20,20)
	frc_x = ax.bar(range(N), forces[0], width, color='b')
	frc_y = ax.bar(np.arange(N)+width, forces[1], width, color='g')
	#fig.canvas.draw()

	plt.show()
	"""
