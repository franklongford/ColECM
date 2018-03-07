import numpy as np
import os
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

""" STANDARD ROUTINES """

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)


def numpy_remove(list1, list2):
	"""
	numpy_remove(list1, list2)
	Deletes overlapping elements of list2 from list1
	"""

	return np.delete(list1, np.where(np.isin(list1, list2)))


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


def get_dx_dy_dz(pos, N, cell_dim):
	"""
	get_dx_dy_dz(pos, N, cell_dim)

	Calculate distance vector between two beads
	"""

	N = pos.shape[0]
	temp_pos = np.moveaxis(pos, 0, 1)

	dx = np.tile(temp_pos[0], (N, 1))
	dy = np.tile(temp_pos[1], (N, 1))
	dz = np.tile(temp_pos[2], (N, 1))

	dx = dx.T - dx
	dy = dy.T - dy
	dz = dz.T - dz

	dx -= cell_dim[0] * np.array(2 * dx / cell_dim[0], dtype=int)
	dy -= cell_dim[1] * np.array(2 * dy / cell_dim[1], dtype=int)
	dz -= cell_dim[2] * np.array(2 * dz / cell_dim[2], dtype=int)

	return dx, dy, dz


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

	bond_beads = []
	dxdydz_index = []
	r_index = []

	count = np.unique(bond_index_full.T[0]).shape[0]

	for n in range(N):
		slice_full = np.argwhere(bond_index_full.T[0] == n)
		slice_half = np.argwhere(bond_index_half.T[0] == n)

		if slice_full.shape[0] > 1:
			bond_beads.append(np.unique(bond_index_full[slice_full].flatten()))
			dxdydz_index.append(bond_index_full[slice_full])
			r_index.append(slice_full[0])

	bond_beads = np.array(bond_beads)
	dxdy_index = np.reshape(dxdydz_index, (2 * len(dxdydz_index), 2))
	r_index = np.array([np.argwhere(np.sum(bond_index_half**2, axis=1) == x).flatten() for x in np.sum(dxdydz_index**2, axis=1)]).flatten()

	return bond_beads, dxdydz_index, r_index


def calc_energy_forces(dx, dy, dz, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxdydz_index, r_index):
	"""
	calc_energy_forces_linear(dx, dy, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc)

	Return tot potential energy and forces on each bead in simulation
	"""

	N = dx.shape[0]
	f_beads_x = np.zeros((N))
	f_beads_y = np.zeros((N))
	f_beads_z = np.zeros((N))
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

		r_half = np.sqrt(r2[indices_half])
		r_full = np.repeat(r_half, 2)
		
		bond_pot = pot_harmonic(r_half, bond_param[0], bond_param[1])
		pot_energy += np.sum(bond_pot)

		bond_frc = force_harmonic(r_half, bond_param[0], bond_param[1])
		for i, sign in enumerate([1, -1]):
			f_beads_x[indices_half[i]] += sign * (bond_frc * dx[indices_half] / r_half)
			f_beads_y[indices_half[i]] += sign * (bond_frc * dy[indices_half] / r_half)
			f_beads_z[indices_half[i]] += sign * (bond_frc * dz[indices_half] / r_half)

		"Bond Angles"

		try:
			"Make array of vectors rij, rjk for all connected bonds"
			vector = np.stack((dx[create_index(dxdy_index)], dy[create_index(dxdy_index)]), axis=1)
			n_vector = int(vector.shape[0])

			"Find |rij| values for each vector"
			r_vector = r_half[r_index]
			"Calculate |rij||rjk| product for each pair of vectors"
			r_prod = np.prod(np.reshape(r_vector, (int(n_vector/2), 3)), axis = 1)

			"Form dot product of each vector pair rij*rjk in vector array corresponding to an angle"
			dot_prod = np.sum(np.prod(np.reshape(vector, (int(n_vector/2), 2, 3)), axis=1), axis=1)
			"Calculate cos(theta) for each angle"
			cos_the = dot_prod / r_prod

			"Form arrays of |rij| vales, cos(theta) and |rij||rjk| terms same shape as vector array"
			r_array = np.reshape(np.repeat(r_vector, 2), vector.shape)
			cos_the_array = np.reshape(np.repeat(cos_the, 4), vector.shape)
			r_prod_array = np.reshape(np.repeat(r_prod, 4), vector.shape)

			"Form left and right hand side terms of (cos(theta) rij / |rij|^2 - rjk / |rij||rjk|)"
			r_left = cos_the_array * vector / r_array**2
			r_right = vector / r_prod_array

			ij_indices = np.arange(0, vector.shape[0], 2)
			jk_indices = np.arange(1, vector.shape[0], 2)

			"Perfrom right hand - left hand term for every rij rkj pair"
			r_left[ij_indices] -= r_right[jk_indices]
			r_left[jk_indices] -= r_right[ij_indices] 
		
			"Calculate forces upon beads i, j and k"
			frc_angle_ij = angle_param[1] * r_left
			frc_angle_k = -np.sum(np.reshape(frc_angle_ij, (int(n_vector/2), 2, 3)), axis=1) 

			"Add angular forces to force array" 
			f_beads_x[bond_beads.T[0]] -= frc_angle_ij[ij_indices].T[0]
			f_beads_x[bond_beads.T[1]] -= frc_angle_k.T[0]
			f_beads_x[bond_beads.T[2]] -= frc_angle_ij[jk_indices].T[0]

			f_beads_y[bond_beads.T[0]] -= frc_angle_ij[ij_indices].T[1]
			f_beads_y[bond_beads.T[1]] -= frc_angle_k.T[1]
			f_beads_y[bond_beads.T[2]] -= frc_angle_ij[jk_indices].T[1]

			f_beads_z[bond_beads.T[0]] -= frc_angle_ij[ij_indices].T[2]
			f_beads_z[bond_beads.T[1]] -= frc_angle_k.T[2]
			f_beads_z[bond_beads.T[2]] -= frc_angle_ij[jk_indices].T[2]

		except IndexError: pass


	non_zero = np.nonzero(r2)
	nonbond_pot = pot_vdw(r2[non_zero] * verlet_list[non_zero], vdw_param[0], vdw_param[1]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	nonbond_frc = force_vdw(r2 * verlet_list, vdw_param[0], vdw_param[1]) - cut_frc
	f_beads_x += np.nansum(nonbond_frc * dx / r2, axis=0)
	f_beads_y += np.nansum(nonbond_frc * dy / r2, axis=0)
	f_beads_z += np.nansum(nonbond_frc * dz / r2, axis=0)

	frc_beads = np.transpose(np.array([f_beads_x, f_beads_y, f_beads_z]))

	return pot_energy, frc_beads

""" Molecular Mechanics Verlocity Verlet Integration """


def grow_fibre(n, bead, N, pos, bond_matrix, vdw_param, bond_param, angle_param, rc, cell_dim, max_energy):
	"""
	grow_fibre(n, bead, N, pos, bond_matrix, vdw_param, bond_param, angle_param, rc, cell_dim, max_energy)

	Grow collagen fibre consisting of beads
	"""

	if bead == 0:
		pos[n] = np.random.random((3)) * vdw_param[0] * 2

	else:
		bond_matrix[n][n-1] = 1
		bond_matrix[n-1][n] = 1

		bond_beads, dxdy_index, r_index = update_bond_lists(bond_matrix)

		energy = max_energy + 1

		while energy > max_energy:
			new_vec = rand_vector(3) * vdw_param[0]
			pos[n] = pos[n-1] + new_vec
			dx, dy, dz = get_dx_dy_dz(np.array(pos), N, cell_dim)
			r2 = dx**2 + dy**2 + dz**2

			energy, _ = calc_energy_forces(dx, dy, dz, r2, bond_matrix, check_cutoff(r2, rc**2), vdw_param, bond_param, angle_param, rc, bond_beads, dxdy_index, r_index)
			
	return pos, bond_matrix


def setup(file_name, cell_dim, n_fibre, l_fibre, mass, kBT, vdw_param, bond_param, angle_param, rc):
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
	pos = np.zeros((N, 3), dtype=float)
	bond_matrix = np.zeros((N, N), dtype=int)

	if not os.path.exists(file_name):

		print("Growing {} fibres containing {} beads".format(n_fibre, l_fibre))

		for bead in range(l_fibre):
			pos, bond_matrix = grow_fibre(bead, bead, N, pos, bond_matrix, vdw_param, bond_param, 
						      angle_param, rc, cell_dim, 1E2)

		pos -= np.min(pos)

		size_x = np.max(pos.T[0]) + vdw_param[0] * 3
		size_y = np.max(pos.T[1]) + vdw_param[0] * 3
		size_z = np.max(pos.T[2]) + vdw_param[0] * 3
		bead_list = np.arange(0, l_fibre)

		for fibre in range(1, n_fibre):
		
			pos_x = pos.T[0][bead_list] + size_x * int(fibre / np.sqrt(n_fibre))
			pos_y = pos.T[1][bead_list] + size_y * int(fibre % np.sqrt(n_fibre))
			pos_z = pos.T[2][bead_list] + size_z * int(fibre % np.sqrt(n_fibre))

			pos[bead_list + l_fibre * fibre] += np.array((pos_x, pos_y, pos_z)).T

			for bead in range(1, l_fibre):
				n = fibre * l_fibre + bead
				bond_matrix[n][n-1] = 1
				bond_matrix[n-1][n] = 1

		cell_dim = np.array([np.max(pos.T[0]) + vdw_param[0], np.max(pos.T[1]) + vdw_param[0], np.max(pos.T[2]) + vdw_param[0]])

	else:
		print("Loading restart file {}".format(file_name))

		pos = np.load(file_name)
		cell_dim = pos[-1]
		pos = pos[:-1]

		for fibre in range(n_fibre):
			for bead in range(1, l_fibre):
					n = fibre * l_fibre + bead
					bond_matrix[n][n-1] = 1
					bond_matrix[n-1][n] = 1


	vel = (np.random.random((N, 3)) - 0.5) * 2 * kBT
	dx, dy, dz = get_dx_dy_dz(pos, N, cell_dim)
	r2 = dx**2 + dy**2 + dz**2

	verlet_list = check_cutoff(r2, rc**2)

	bond_beads, dxdydz_index, r_index = update_bond_lists(bond_matrix)
	_, frc = calc_energy_forces(dx, dy, dz, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxdy_index, r_index)

	return pos, vel, frc, cell_dim, bond_matrix, verlet_list, bond_beads, dxdydz_index, r_index


def velocity_verlet_alg(pos, vel, frc, mass, bond_matrix, verlet_list, bond_beads, dxdy_index, r_index, dt, cell_dim, vdw_param, bond_param, angle_param, rc, kBT, gamma, sigma, xi = 0, theta = 0, Langevin=False):
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
 
	dx, dy, dz = get_dx_dy_dz(pos, N, cell_dim)
	r2 = dx**2 + dy**2 + dz**2

	verlet_list = check_cutoff(r2, rc**2)
	pot_energy, new_frc = calc_energy_forces(dx, dy, dz, r2, bond_matrix, verlet_list, vdw_param, bond_param, angle_param, rc, bond_beads, dxdydz_index, r_index)
	vel += 0.5 * dt * new_frc / mass

	if Langevin: vel += 0.5 * dt * frc / mass - gamma * (dt * vel + C) + sigma * xi * np.sqrt(dt)
	
	frc = new_frc
	tot_energy = pot_energy + kin_energy(vel)

	return pos, vel, frc, verlet_list, tot_energy
