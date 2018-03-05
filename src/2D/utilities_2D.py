"""
COLLAGEN FIBRIL SIMULATION 2D

Created by: Frank Longford
Created on: 01/11/15

Last Modified: 02/02/2018
"""

import numpy as np
import scipy.constants as con
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
import random

import sys
import os


""" STANDARD ROUTINES """


def unit_vector(vector):

	sum_ = np.sum([i**2 for i in vector])
	norm = 1./sum_
	return np.array([np.sqrt(i**2 * norm) * np.sign(i) for i in vector])


def rand_vector(n): return unit_vector(np.random.random((n)) * 2 - 1) 


def remove_element(a, array): return np.array([x for x in array if x != a])


""" Molecular Mechanics Verlocity Verlet Integration """

def setup(boxl, nchain, lchain, kBT, vdw_param, bnd_param, angle_param, rc):

	N = nchain*lchain
	pos = []
	bond_atom = np.zeros((N, N), dtype=int)
	mass = np.ones(N)

	if nchain > 1: n_section = np.sqrt(np.min([i for i in np.arange(nchain+1)**2 if i >= nchain]))
	else: n_section = 1

	sections = np.arange(n_section**2)

	for chain in range(nchain):
		section = random.choice(sections)
		sections = remove_element(section, sections)

		lim_x = boxl / n_section * (section % n_section)
		lim_y = boxl / n_section * int(section / n_section)

		for bead in range(lchain):
			n = chain * lchain + bead
			pos, bond_atom, bond_angle = grow_chain(bead, n, N, pos, bond_atom, vdw_param, bnd_param, angle_param, rc, boxl, n_section, lim_x, lim_y, 1E2)

	pos = np.array(pos)

	vel = (np.random.random((N,2)) - 0.5) * 2 * kBT
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2
	verlet_list = check_cutoff(r2, rc**2)

	frc = calc_forces(N, boxl, dx, dy, r2, bond_atom, bond_angle, verlet_list, vdw_param, bnd_param, angle_param, rc)

	return pos, vel, frc, bond_atom, bond_angle, boxl, mass


def grow_chain(bead, n, N, pos, bond_atom, vdw_param, bnd_param, angle_param, rc, boxl, n_section, lim_x, lim_y, max_energy):

	print(bead, n)

	if bead == 0:
		pos.append(np.random.random((2)) * boxl / n_section + np.array([lim_x, lim_y]))
		bond_angle = 0

	else:
		bond_atom[n][n-1] = 1
		bond_atom[n-1][n] = 1

		nbond = np.sum(np.tril(bond_atom))
		bond_angle = np.zeros((nbond, nbond))
		bond_index = np.argwhere(np.tril(bond_atom))
		for i in range(nbond):
			for j in range(i):
				bond_angle[i][j] = np.in1d(bond_index[i], bond_index[j]).any()
				bond_angle[j][i] += bond_angle[i][j]

		new_vec = rand_vector(2) * vdw_param[0]
		pos.append(pos[n-1] + new_vec)
		dx, dy = get_dx_dy(np.array(pos), N, boxl)
		r2 = dx**2 + dy**2
		energy = pot_energy(dx, dy, r2, bond_atom, bond_angle, boxl, vdw_param, bnd_param, angle_param, rc)

		while energy > max_energy:
			new_vec = rand_vector(2) * vdw_param[0]
			pos[-1] = pos[-2] + new_vec
			dx, dy = get_dx_dy(np.array(pos), N, boxl)
			r2 = dx**2 + dy**2
			energy = pot_energy(dx, dy, r2, bond_atom, bond_angle, boxl, vdw_param, bnd_param, angle_param, rc)
			
	return pos, bond_atom, bond_angle


def get_dx_dy(pos, N, boxl):

	N = pos.shape[0]
	temp_pos = np.moveaxis(pos, 0, 1)

	dx = np.tile(temp_pos[0], (N, 1))
	dy = np.tile(temp_pos[1], (N, 1))

	dx = dx.T - dx
	dy = dy.T - dy

	dx -= boxl * np.array(2 * dx / boxl, dtype=int)
	dy -= boxl * np.array(2 * dy / boxl, dtype=int)

	return dx, dy


def check_cutoff(array, rc):

	return (array <= rc).astype(float)


def calc_forces(N, boxl, dx, dy, r2, bond_atom, bond_angle, verlet_list, vwd_param, bnd_param, angle_param, rc):

	f_beads_x = np.zeros((N))
	f_beads_y = np.zeros((N))
	cut_frc = force_vdw(rc**2, vwd_param[0], vwd_param[1])

	nbond = int(np.sum(np.tril(bond_atom)))

	if nbond > 0:
		"Bond Lengths"
		bond_index_half = np.argwhere(np.triu(bond_atom))
		indices_half = create_index(bond_index_half)
		bond_index_full = np.argwhere(bond_atom)
		indices_half = create_index(bond_index_full)

		r = np.sqrt(r2[indices_half])
		bond_frc = force_harmonic(r, bnd_param[0], bnd_param[1])

		for i, sign in enumerate([1, -1]):
			f_beads_x[indices_half[i]] += sign * (bond_frc * dx[indices_half] / r)
			f_beads_y[indices_half[i]] += sign * (bond_frc * dy[indices_half] / r)

		#"""
		if np.sum(bond_angle) > 0:
			"Bond Angles"
			count = np.unique(bond_index_full.T[0]).shape[0]
			r = np.repeat(r, 2)

			for n in range(count):
				slice_full = np.argwhere(bond_index_full.T[0] == n)
				slice_half = np.argwhere(bond_index_half.T[0] == n)

				nb = slice_full.shape[0]

				if nb > 1:
					atoms = np.unique(bond_index_full[slice_full].flatten())
					switch = np.ones((nb, nb)) - np.identity(nb)
					indices_full = create_index(bond_index_full[slice_full])

					vector = np.concatenate((dx[indices_full], dy[indices_full]), axis=0).T
					vector_matrix = np.reshape(np.tile(vector, (1, nb)), (nb, nb, 2))
					r_matrix = np.reshape(np.tile(r[slice_full].flatten(), (1, nb)), (nb, nb)).T

					dot_prod = np.sum(vector_matrix * np.moveaxis(vector_matrix, 0, 1), axis=2)
					cos_the = dot_prod / (r_matrix * r_matrix.T)
					vector_norm = vector / r[slice_full]
						
					frc_angle = angle_param[1] * (vector_norm * cos_the[0][1] - np.flip(vector_norm, axis=0)) / r[slice_full]
					frc_angle = np.insert(frc_angle, -1, -np.sum(frc_angle, axis=0), axis=0)
		
					f_beads_x[atoms] -= frc_angle.T[0]
					f_beads_y[atoms] -= frc_angle.T[1]
	
		#"""

	nonbond_frc = force_vdw(r2 * verlet_list , vwd_param[0], vwd_param[1]) - cut_frc

	f_beads_x -= np.nansum(nonbond_frc * dx / r2, axis=0)
	f_beads_y -= np.nansum(nonbond_frc * dy / r2, axis=0)

	f_beads = np.transpose(np.array([f_beads_x, f_beads_y]))

	return f_beads


def create_index(array):
    
    return (np.array(array.T[0]), np.array(array.T[1]))


def pot_energy(dx, dy, r2, bond_atom, bond_angle, boxl, vdw_param, bnd_param, angle_param, rc):

	N = dx.shape[0]
	cut_pot = pot_vdw(rc**2, vdw_param[0], vdw_param[1])
	pot_energy = 0

	nbond = int(np.sum(np.tril(bond_atom)))

	if nbond > 0:
		bond_index_half = np.argwhere(np.triu(bond_atom))
		bond_index_full = np.argwhere(bond_atom)

		r = np.sqrt(r2[create_index(bond_index_half)])

		bond_pot = pot_harmonic(r, bnd_param[0], bnd_param[1])
		pot_energy += np.sum(bond_pot)

		if np.sum(bond_angle) > 0:
			"Bond Angles"
			count = np.unique(bond_index_full.T[0]).shape[0]
			r = np.repeat(r, 2)

			for n in range(count):
				slice_full = np.argwhere(bond_index_full.T[0] == n)
				slice_half = np.argwhere(bond_index_half.T[0] == n)

				nb = slice_full.shape[0]

				if nb > 1:
					switch = np.ones((nb, nb)) - np.identity(nb)
					indices_full = create_index(bond_index_full[slice_full])

					vector = np.array((dx[indices_full], dy[indices_full])).T
					vector_matrix = np.reshape(np.tile(vector, (1, nb)), (nb, nb, 2))
					r_matrix = np.reshape(np.tile(r[slice_full].flatten(), (1, nb)), (nb, nb)).T

					dot_prod = np.sum(vector_matrix * np.moveaxis(vector_matrix, 0, 1), axis=2)
					cos_the = dot_prod / (r_matrix * r_matrix.T)

					pot_energy += np.nansum(np.triu(cos_the) * angle_param[1] * switch)

	nonbond_pot = pot_vdw(r2 * check_cutoff(r2, rc**2), vdw_param[0], vdw_param[1]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	return pot_energy


def tot_energy(N, dx, dy, r2, vel, bond_atom, bond_angle, boxl, vdw_param, bnd_param, angle_param, rc):

	tot_energy = 0

	tot_energy += pot_energy(dx, dy, r2, bond_atom, bond_angle, boxl, vdw_param, bnd_param, angle_param, rc)
	tot_energy += kin_energy(vel)
	
	return tot_energy 


SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)

def VV_alg(n, pos, vel, frc, bond_atom, bond_angle, mass, verlet_list, dt, boxl, vdw_param, bnd_param, angle_param, rc, kBT, gamma, sigma, xi = 0, theta = 0, Langevin=False):

	N = pos.shape[0]
	
	vel += 0.5 * dt * frc / np.array([mass, mass]).T
	pos += dt * vel

	if Langevin:
		C = 0.5 * dt**2 * (frc / np.array([mass, mass]).T - gamma * vel) + sigma * dt**(3./2) * (xi + theta / SQRT3) / 2.
		pos += C

	#pos += boxl * (1 - np.array((pos + boxl) / boxl, dtype=int))
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2

	verlet_list = check_cutoff(r2, rc**2)
	new_frc = calc_forces(N, boxl, dx, dy, r2, bond_atom, bond_angle, verlet_list, vdw_param, bnd_param, angle_param, rc)
	vel += 0.5 * dt * new_frc / np.array([mass, mass]).T

	if Langevin: vel += 0.5 * dt * frc / np.array([mass, mass]).T - gamma * (dt * vel + C) + sigma * xi * np.sqrt(dt)
	
	frc = new_frc
	energy = tot_energy(N, dx, dy, r2, vel, bond_atom, bond_angle, boxl, vdw_param, bnd_param, angle_param, rc)

	return pos, vel, frc, verlet_list, energy



def pot_morse(x, x0, k, D): return D * (1 - np.exp(- k * (x - x0)))**2


def force_harmonic(x, x0, k): return 2 * k * (x0 - x)


def pot_harmonic(x, x0, k): return k * (x - x0)**2


def force_sine(x, x0, k): return k * np.cos(x0 - x)


def pot_sine(x, x0, k): return - k * np.sin(x0 - x)


def force_vdw(r2, sig1, ep1): return 24 * ep1 * (2 * (sig1/r2)**6 - (sig1/r2)**3)


def pot_vdw(r2, sig1, ep1): return 4 * ep1 * ((sig1**12/r2**6) - (sig1**6/r2**3))


def kin_energy(vel): return np.mean(vel**2)




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
