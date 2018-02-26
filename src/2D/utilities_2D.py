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

def setup(boxl, nchain, lchain, kBT, vdw_param, bnd_param, rc):

	N = nchain*lchain
	pos = np.zeros((N, 2))
	atom_bonds = np.zeros((N, N), dtype=int)
	mass = np.ones(N)

	if nchain > 1: n_section = np.sqrt(np.min([i for i in np.arange(nchain+1)**2 if i >= nchain]))
	else: n_section = 1

	print(n_section)

	sections = np.arange(n_section**2)

	for chain in range(nchain):
		section = random.choice(sections)
		sections = remove_element(section, sections)

		lim_x = boxl / n_section * (section % n_section)
		lim_y = boxl / n_section * int(section / n_section)

		for bead in range(lchain):
			i = chain * lchain + bead
			pos, atom_bonds = grow_chain(bead, i, N, pos, vdw_param, bnd_param, rc, atom_bonds, boxl, n_section, lim_x, lim_y, 1E3)

	boxl = np.max(pos)
	pos += boxl * (1 - np.array((pos + boxl) / boxl, dtype=int))

	vel = (np.random.random((N,2)) - 0.5) * 2 * kBT
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2
	verlet_list = check_cutoff(r2, rc**2)
	frc = calc_forces(N, boxl, dx, dy, r2, atom_bonds, verlet_list, vdw_param, bnd_param, rc)

	return pos, vel, frc, atom_bonds, boxl, mass


def grow_chain(bead, i, N, pos, vdw_param, bnd_param, rc, bond, boxl, n_section, lim_x, lim_y, max_energy):

	if bead == 0:
		pos[i] = np.random.random((2)) * boxl / n_section + np.array([lim_x, lim_y])
	else:
		energy = max_energy + 1
		while  energy > max_energy:
			new_vec = rand_vector(2) * vdw_param[0]
			pos[i] = pos[i-1] + new_vec
			dx, dy = get_dx_dy(pos, N, boxl)
			r2 = dx**2 + dy**2	
			energy = pot_energy(dx, dy, r2, bond, boxl, vdw_param, bnd_param, rc)
			
		pos[i] += boxl * (1 - np.array((pos[i] + boxl) / boxl, dtype=int))

		bond[i][i-1] = 1
		bond[i-1][i] = 1

	return pos, bond


def get_dx_dy(pos, N, boxl):

	temp_pos = np.moveaxis(pos, 0, 1)

	dx = np.tile(temp_pos[0], (N, 1))
	dy = np.tile(temp_pos[1], (N, 1))

	dx -= np.transpose(dx)
	dy -= np.transpose(dy)

	dx -= boxl * np.array(2 * dx / boxl, dtype=int)
	dy -= boxl * np.array(2 * dy / boxl, dtype=int)

	return dx, dy


def check_cutoff(array, rc):

	return (array <= rc).astype(float)


def calc_forces(N, boxl, dx, dy, r2, bond, verlet_list, vwd_param, bnd_param, rc):

	f_beads_x = np.zeros((N))
	f_beads_y = np.zeros((N))
	cut_frc = force_vdw(rc**2, vwd_param[0], vwd_param[1])

	if np.sum(bond) > 0:
		bond_index = np.where(np.tril(bond))
		r = np.sqrt(r2[bond_index])
		bond_frc = force_bond(r, bnd_param[0], bnd_param[1])
		for i, sign in enumerate([-1, 1]): 
			f_beads_x[bond_index[i]] += sign * (bond_frc * dx[bond_index] / r)
			f_beads_y[bond_index[i]] += sign * (bond_frc * dy[bond_index] / r)

	nonbond_frc = force_vdw((r2 - bond * r2) * verlet_list , vwd_param[0], vwd_param[1]) - cut_frc
	f_beads_x += np.nansum(nonbond_frc * dx / r2, axis=0)
	f_beads_y += np.nansum(nonbond_frc * dy / r2, axis=0)

	f_beads = np.transpose(np.array([f_beads_x, f_beads_y]))

	return f_beads

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)

def VV_alg(n, pos, vel, frc, atom_bonds, mass, verlet_list, dt, boxl, vdw_param, bnd_param, rc, kBT, gamma, sigma, xi = 0, theta = 0, Langevin=False):

	N = pos.shape[0]
	
	vel += 0.5 * dt * frc / np.array([mass, mass]).T
	pos += dt * vel

	if Langevin:
		C = 0.5 * dt**2 * (frc / np.array([mass, mass]).T - gamma * vel) + sigma * dt**(3./2) * (xi + theta / SQRT3) / 2.
		pos += C

	pos += boxl * (1 - np.array((pos + boxl) / boxl, dtype=int))
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2

	verlet_list = check_cutoff(r2, rc**2)
	new_frc = calc_forces(N, boxl, dx, dy, r2, atom_bonds, verlet_list, vdw_param, bnd_param, rc)
	vel += 0.5 * dt * new_frc / np.array([mass, mass]).T

	if Langevin: vel += 0.5 * dt * frc / np.array([mass, mass]).T - gamma * (dt * vel + C) + sigma * xi * np.sqrt(dt)
	
	frc = new_frc
	energy = tot_energy(N, dx, dy, r2, vel, atom_bonds, boxl, vdw_param, bnd_param, rc)

	return pos, vel, frc, verlet_list, energy


def force_bond(r, r0, kB): return 2 * kB * (r0 - r)


def force_angle(theta, theta0, kA): return 2 * kA * (theta0 - theta)


def force_vdw(r2, sig1, ep1): return 24 * ep1 * (2 * (sig1/r2)**6 - (sig1/r2)**3)


def pot_vdw(r2, sig1, ep1): return 4 * ep1 * ((sig1**12/r2**6) - (sig1**6/r2**3))


def pot_bond(r, r0, kB): return kB * (r - r0)**2


def pot_angle(theta, theta0, kA): return kA * (theta - theta0)**2


def kin_energy(vel): return np.mean(vel**2)


def pot_energy(dx, dy, r2, bond, boxl, vdw_param, bnd_param, rc):

	N = dx.shape[0]
	cut_pot = pot_vdw(rc**2, vdw_param[0], vdw_param[1])
	pot_energy = 0

	if np.sum(bond) > 0:
		bond_index = np.where(np.tril(bond))
		r = np.sqrt(r2[bond_index])
		bond_pot = pot_bond(r, bnd_param[0], bnd_param[1])
		pot_energy += np.sum(bond_pot)

	nonbond_pot = pot_vdw((r2 - bond * r2) * check_cutoff(r2, rc**2), vdw_param[0], vdw_param[1]) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	return pot_energy


def tot_energy(N, dx, dy, r2, vel, bond, boxl, vdw_param, bnd_param, rc):

	tot_energy = 0

	tot_energy += pot_energy(dx, dy, r2, bond, boxl, vdw_param, bnd_param, rc)
	tot_energy += kin_energy(vel)
	
	return tot_energy 


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
