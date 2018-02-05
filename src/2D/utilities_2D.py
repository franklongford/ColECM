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

def setup(boxl, nchain, lchain, sig1, ep1, r0, kB, rc):

	N = nchain*lchain
	pos = np.zeros((N, 2))
	bond = np.zeros((N, N))
	if nchain > 1: n_section = np.sqrt(np.min([i for i in np.arange(nchain)**2 if i >= nchain]))
	else: n_section = 1
	sections = np.arange(n_section**2)

	for chain in range(nchain):
		section = random.choice(sections)
		sections = remove_element(section, sections)

		lim_x = boxl / n_section * (section % n_section)
		lim_y = boxl / n_section * int(section / n_section)

		for bead in range(lchain):
			i = chain * lchain + bead
			pos, bond = grow_chain(bead, i, N, pos, sig1, ep1, r0, kB, rc, bond, boxl, n_section, lim_x, lim_y, 1E3)

	vel = (np.random.random((N,2)) - 0.5) * 2
	frc, _, _, _ = calc_forces(N, boxl, pos, bond, sig1, ep1, r0, kB, rc)

	return pos, vel, frc, bond


def grow_chain(bead, i, N, pos, sig1, ep1, r0, kB, rc, bond, boxl, n_section, lim_x, lim_y, max_energy):

	if bead == 0:
		pos[i] = np.random.random((2)) * boxl / n_section + np.array([lim_x, lim_y])
	else:
		energy = max_energy + 1
		while  energy > max_energy:
			pos[i] = pos[i-1] + rand_vector(2) * sig1
			energy = tot_energy(N, pos, bond, boxl, sig1, ep1, r0, kB, rc)
			
		for n in range(2): pos[i][n] += boxl * (1 - int((pos[i][n] + boxl) / boxl))
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
	

def calc_forces(N, boxl, pos, bond, sig1, ep1, r0, kB, rc):

	f_beads = np.zeros((N, 2))
	cut_frc = force_vdw(rc**2, sig1, ep1)
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2

	for i in range(N):
		for j in range(i):
			k = i
			#dx = (pos[i][0] - pos[j][0])
			#dx -= boxl * int(2*dx/boxl)
			#dy = (pos[i][1] - pos[j][1])
			#dy -= boxl * int(2*dy/boxl)
			#r2 = dx**2 + dy**2

			if bond[i][j] == 1:
				r = np.sqrt(r2[i][j])
				Fr = force_bond(r, r0, kB)
				f_beads[i][0] -= dx[i][j] / r * Fr
				f_beads[i][1] -= dy[i][j] / r * Fr

				f_beads[j][0] += dx[i][j] / r * Fr
				f_beads[j][1] += dy[i][j] / r * Fr

			else:
				if r2[i][j] <= rc**2:
					Fr = force_vdw(r2[i][j], sig1, ep1) - cut_frc
					f_beads[i][0] -= dx[i][j] / r2[i][j] * Fr
					f_beads[i][1] -= dy[i][j] / r2[i][j] * Fr

					f_beads[j][0] += dx[i][j] / r2[i][j] * Fr
					f_beads[j][1] += dy[i][j] / r2[i][j] * Fr

			#print "{} {} {}".format(x, y, r)

	return f_beads, dx, dy, r2


def VV_alg(pos, vel, frc, bond, dt, N, boxl, sig1, ep1, r0, kB, rc):

	for i in range(N):
		for j in range(2):  
			vel[i][j] += 0.5 * dt * frc[i][j]
			pos[i][j] += dt * vel[i][j]
			pos[i][j] += boxl * (1 - int((pos[i][j] + boxl) / boxl))

	frc, dx, dy, r2 = calc_forces(N, boxl, pos, bond, sig1, ep1, r0, kB, rc)

	for i in range(N): 
		for j in range(2): vel[i][j] += 0.5 * dt * frc[i][j]

	return pos, vel, frc


def force_bond(r, r0, kB): return 2 * kB * (r0 - r)


def force_vdw(r2, sig1, ep1): return 24 * ep1 * (2 * (sig1/r2)**6 - (sig1/r2)**3)


def pot_vdw(r2, sig1, ep1): return 4 * ep1 * ((sig1**12/r2**6) - (sig1**6/r2**3))


def pot_bond(r, r0, kB): return kB * (r - r0)**2


def tot_energy(N, pos, bond, boxl, sig1, ep1, r0, kB, rc):

	energy = 0
	cut_energy = pot_vdw(rc**2, sig1, ep1)
	for i in range(N):
		for j in range(i):
			if np.dot(pos[i], pos[j]) != 0:
				dx = (pos[i][0] - pos[j][0])
				dx -= boxl * int(2*dx/boxl)
				dy = (pos[i][1] - pos[j][1])
				dy -= boxl * int(2*dy/boxl)

				r2 = dx**2 + dy**2

				if bond[i][j] == 1:
					r = np.sqrt(r2)
					energy += pot_bond(r, r0, kB)

				elif r2 <= rc**2: energy += pot_vdw(r2, sig1, ep1) - cut_energy
	return energy 


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
