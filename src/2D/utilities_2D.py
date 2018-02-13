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

def setup(boxl, nchain, lchain, T, sig1, ep1, r0, kB, rc, Sdiag):

	N = nchain*lchain
	pos = np.zeros((N, 2))
	bond = np.zeros((N, N), dtype=int)

	if nchain > 1: n_section = np.sqrt(np.min([i for i in np.arange(nchain+1)**2 if i >= nchain]))
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

	boxl = np.max(pos)
	pos += boxl * (1 - np.array((pos + boxl) / boxl, dtype=int))

	con_index, con_coeff, atom1, atom2, ncc = setup_lincs(nchain, lchain, bond, 1, Sdiag)

	vel = (np.random.random((N,2)) - 0.5) * 2 * T
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2
	verlet_list = check_cutoff(r2, rc**2)
	frc = calc_forces(N, boxl, dx, dy, r2, bond, verlet_list, sig1, ep1, r0, kB, rc)

	return pos, vel, frc, bond, boxl, con_index, con_coeff, atom1, atom2, ncc


def grow_chain(bead, i, N, pos, sig1, ep1, r0, kB, rc, bond, boxl, n_section, lim_x, lim_y, max_energy):

	if bead == 0:
		pos[i] = np.random.random((2)) * boxl / n_section + np.array([lim_x, lim_y])
	else:
		energy = max_energy + 1
		while  energy > max_energy:
			new_vec = rand_vector(2) * sig1
			pos[i] = pos[i-1] + new_vec
			dx, dy = get_dx_dy(pos, N, boxl)
			r2 = dx**2 + dy**2	
			energy = pot_energy(dx, dy, r2, bond, boxl, sig1, ep1, r0, kB, rc)
			
		pos[i] += boxl * (1 - np.array((pos[i] + boxl) / boxl, dtype=int))

		bond[i][i-1] = 1
		bond[i-1][i] = 1

	return pos, bond


def setup_2(boxl, N, T, sig1, ep1, r0, kB, rc):

	pos = np.zeros((N, 2))
	bond = np.zeros((N, N))
	beads = np.arange(N)

	i = 0
	chain = 0
	bead = 0
	growing = True
	loop = 0

	while growing:
		sys.stdout.write("Placing bead {}/{}  chain {}   box = {}\r".format(i, N, chain, boxl))
		sys.stdout.flush()
		pos, bond, check = grow_chain_2(bead, i, N, pos, sig1, ep1, r0, kB, rc, bond, boxl, 1E3)

		if check:
			i += 1
			bead += 1
			loop = 0
		else: loop += 1

		if np.random.random() > 0.9 or loop > 10:
			chain += 1
			bead = 0
			loop = 0

		if i >= N: growing = False

	vel = (np.random.random((N,2)) - 0.5) * 2 * T
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2
	verlet_list = check_cutoff(r2, rc**2)
	frc = calc_forces(N, boxl, dx, dy, r2, bond, verlet_list, sig1, ep1, r0, kB, rc)

	return pos, vel, frc, bond


def setup_3(boxl, nchain, lchain, T, sig1, ep1, r0, kB, rc):

	N = nchain*lchain
	pos = np.zeros((N, 2))
	bond = np.zeros((N, N))

	if nchain > 1: n_section = np.sqrt(np.min([i for i in np.arange(nchain+1)**2 if i >= nchain]))
	else: n_section = 1

	sections = np.arange(n_section**2)

	for chain in range(nchain):
		if chain == 0:
			for bead in range(lchain):
				i = chain * lchain + bead
				pos, bond = grow_chain(bead, i, N, pos, sig1, ep1, r0, kB, rc, bond, boxl, n_section, 0, 0, 1E2)

			width_x = np.nanmax(np.moveaxis(pos, 0, 1)[0]) - np.nanmin(np.moveaxis(pos, 0, 1)[0])
			width_y = np.nanmax(np.moveaxis(pos, 0, 1)[1]) - np.nanmin(np.moveaxis(pos, 0, 1)[1])

			print(width_x, width_y)

		else:
			for bead in range(lchain):
				i = chain * lchain + bead
				pos[i][0] = pos[bead][0] + width_x * chain
				pos[i][1] = pos[bead][1] + width_y * chain
				if bead != 0:
					bond[i][i-1] = 1
					bond[i-1][i] = 1

	boxl = np.max(pos)
	pos += boxl * (1 - np.array((pos + boxl) / boxl, dtype=int))

	vel = (np.random.random((N,2)) - 0.5) * 2 * T
	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2
	verlet_list = check_cutoff(r2, rc**2)
	frc = calc_forces(N, boxl, dx, dy, r2, bond, verlet_list, sig1, ep1, r0, kB, rc)

	return pos, vel, frc, bond, boxl


def grow_chain_2(bead, i, N, pos, sig1, ep1, r0, kB, rc, bond, boxl, max_energy):

	if bead == 0:
		energy = max_energy + 1
		while  energy > max_energy:
			pos[i] = np.random.random((2)) * boxl
			dx, dy = get_dx_dy(pos, N, boxl)
			r2 = dx**2 + dy**2	
			energy = pot_energy(dx, dy, r2, bond, boxl, sig1, ep1, r0, kB, rc)
	else:
		energy = max_energy + 1
		attempt = 0
		while  energy > max_energy:
			pos[i] = pos[i-1] + rand_vector(2) * sig1
			dx, dy = get_dx_dy(pos, N, boxl)
			r2 = dx**2 + dy**2	
			energy = pot_energy(dx, dy, r2, bond, boxl, sig1, ep1, r0, kB, rc)
			attempt += 1
			if attempt > 100: return pos, bond, False
			
		pos[i] += boxl * (1 - np.array((pos[i] + boxl) / boxl, dtype=int))

		bond[i][i-1] = 1
		bond[i-1][i] = 1

	return pos, bond, True


def setup_lincs(nchain, lchain, bond, m, Sdiag):

	N = nchain*lchain
	K = int(np.sum(bond) / 2)
	cmax = 2

	con_index= np.zeros((K, cmax), dtype=int)
	con_coeff= np.zeros((K, cmax))
	atom1 = np.zeros(K, dtype=int)
	atom2 = np.zeros(K, dtype=int)
	ncc = np.zeros(K, dtype=int)
	kcount = 0

	for i in range(N):
		for j in range(i):
			if bond[i][j] == 1:
				atom1[kcount] = i
				atom2[kcount] = j
				kcount += 1

	for i in range(K):
		if np.remainder(i, lchain-1) == 0: 
			ncc[i] = 1
			con_index[i][0] = i + 1
		elif np.remainder(i, lchain-1) == lchain - 2: 
			ncc[i] = int(1)
			con_index[i][0] = i - 1
		else: 
			ncc[i] = int(2)
			con_index[i][0] = i - 1
			con_index[i][1] = i + 1

		for j in range(ncc[i]):
			print(i, j, atom1[i], atom1[con_index[i][j]], atom2[i], atom2[con_index[i][j]])
			if atom1[i] == atom1[con_index[i][j]] or atom2[i] == atom2[con_index[i][j]]:  
				con_coeff[i][j] = - 1. / m * Sdiag**2
			else:
				con_coeff[i][j] = 1. / m * Sdiag**2

	print(ncc)
	print(con_index)
	#sys.exit()

	return con_index, con_coeff, atom1, atom2, ncc


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


def calc_forces(N, boxl, dx, dy, r2, bond, verlet_list, sig1, ep1, r0, kB, rc):

	f_beads_x = np.zeros((N))
	f_beads_y = np.zeros((N))
	cut_frc = force_vdw(rc**2, sig1, ep1)

	if np.sum(bond) > 0:
		r = np.sqrt(bond * r2)
		bond_frc = force_bond(r, r0, kB) * bond
		f_beads_x += np.nansum(bond_frc * dx / r, axis=0)
		f_beads_y += np.nansum(bond_frc * dy / r, axis=0)

	nonbond_frc = force_vdw((r2 - bond * r2) * verlet_list, sig1, ep1) - cut_frc

	f_beads_x += np.nansum(nonbond_frc * dx / r2, axis=0)
	f_beads_y += np.nansum(nonbond_frc * dy / r2, axis=0)

	f_beads = np.transpose(np.array([f_beads_x, f_beads_y]))

	return f_beads

#def VV_alg(n, pos, vel, frc, bond, verlet_list, dt, N, boxl, sig1, ep1, r0, kB, rc):
def VV_alg(n, pos, vel, frc, bond, verlet_list, dt, nchain, lchain, atom1, atom2, 
			con_index, con_coeff, ncc, boxl, sig1, ep1, r0, kB, rc, Sdiag):

	N = nchain * lchain
	vel += 0.5 * dt * frc

	new_pos = pos + dt * vel
	new_pos += boxl * (1 - np.array((new_pos + boxl) / boxl, dtype=int))

	pos = lincs_np(pos, new_pos, bond, boxl, nchain, lchain, r0, Sdiag, con_index, con_coeff, atom1, atom2, ncc)

	#pos += dt * vel
	pos += boxl * (1 - np.array((pos + boxl) / boxl, dtype=int))

	dx, dy = get_dx_dy(pos, N, boxl)
	r2 = dx**2 + dy**2

	#if n % 5 == 0: verlet_list = check_cutoff(r2, rc**2)
	verlet_list = check_cutoff(r2, rc**2)

	frc = calc_forces(N, boxl, dx, dy, r2, bond, verlet_list, sig1, ep1, r0, kB, rc)

	vel += 0.5 * dt * frc

	energy = tot_energy(N, dx, dy, r2, vel, bond, boxl, sig1, ep1, r0, kB, rc)

	return pos, vel, frc, verlet_list, energy


def force_bond(r, r0, kB): return 2 * kB * (r0 - r)


def force_angle(theta, theta0, kA): return 2 * kA * (theta0 - theta)


def force_vdw(r2, sig1, ep1): return 24 * ep1 * (2 * (sig1/r2)**6 - (sig1/r2)**3)


def pot_vdw(r2, sig1, ep1): return 4 * ep1 * ((sig1**12/r2**6) - (sig1**6/r2**3))


def pot_bond(r, r0, kB): return kB * (r - r0)**2


def pot_angle(theta, theta0, kA): return kA * (theta - theta0)**2


def kin_energy(vel): return np.mean(vel**2)


def pot_energy(dx, dy, r2, bond, boxl, sig1, ep1, r0, kB, rc):

	cut_pot = pot_vdw(rc**2, sig1, ep1)
	pot_energy = 0

	if np.sum(bond) > 0:
		r = np.sqrt(bond * r2)
		bond_pot = pot_bond(r, r0, kB) * bond
		pot_energy += np.nansum(bond_pot) / 2

	nonbond_pot = pot_vdw((r2 - bond * r2) * check_cutoff(r2, rc**2), sig1, ep1) - cut_pot
	pot_energy += np.nansum(nonbond_pot) / 2

	return pot_energy


def tot_energy(N, dx, dy, r2, vel, bond, boxl, sig1, ep1, r0, kB, rc):

	tot_energy = 0

	tot_energy += pot_energy(dx, dy, r2, bond, boxl, sig1, ep1, r0, kB, rc)
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

def lincs_solve(new_pos, K, atom1, atom2, ncc, con_index, Sdiag, B, A, rhs, solution, nrec=2):

	w = 1
	for rec in range(nrec):
		for i in range(K):
			rhs[w,i] = 0
			for n in range(ncc[i]):
				rhs[w,i] = rhs[w,i] + A[i,n] * rhs[2-w, con_index[i,n]]
				solution[i] += rhs[w,i]

	w = 2 - w

	for i in range(K):
		a1 = atom1[i]
		a2 = atom2[i]
		for j in range(2):
			new_pos[a1,j] = new_pos[a1,j] - B[i,j] * Sdiag * solution[i]
			new_pos[a2,j] = new_pos[a2,j] + B[i,j] * Sdiag * solution[i]

	return new_pos


def lincs(old_pos, new_pos, boxl, nchain, lchain, r0, Sdiag, con_index, con_coeff, atom1, atom2, ncc, nrec=2):

	#dx, dy = get_dx_dy(pos, N, boxl)
	#r2 = dx**2 + dy**2
	#r = np.sqrt(r2)

	K = nchain * (lchain-1)
	cmax = lchain-1
	rhs = np.zeros((2, K))
	B = np.zeros((K, 2))
	solution = np.zeros(K)

	for i in range(K):
		B[i]= old_pos[atom1[i]] - old_pos[atom2[i]]
		B[i] -= boxl * np.array(2 * B[i] / boxl, dtype=int)
		r = np.sqrt(np.sum(B[i]**2))
		B[i] *= 1. / r

	A = np.zeros((K, cmax))

	for i in range(K):
		for n in range(ncc[i]):
			k = con_index[i,n]
			A[i,n] = con_coeff[i,n] * (B[i,0]*B[k,0] + B[i,1]*B[k,1])  
			a1 = atom1[i]
			a2 = atom2[i]

			dxy = (new_pos[a1] - new_pos[a2])
			dxy -= boxl * np.array(2 * dxy/ boxl, dtype=int)

			rhs[0,i] = Sdiag * (np.sum(B[i] * dxy) - r0)
			solution[i]=rhs[0,i]

	new_pos = lincs_solve(new_pos, K, atom1, atom2, ncc, con_index, Sdiag, B, A, rhs, solution)

	for i in range(K):
		a1 = atom1[i]
		a2 = atom2[i]

		dxy = (new_pos[a1] - new_pos[a2])
		dxy -= boxl * np.array(2 * dxy/ boxl, dtype=int)

		p = np.sqrt(2 * r0**2 - np.sum(dxy**2))
		rhs[0,i] = Sdiag * (r0 - p)
		solution[i] = rhs[0,i]
 
	new_pos = lincs_solve(new_pos, K, atom1, atom2, ncc, con_index, Sdiag, B, A, rhs, solution)

	return new_pos


def lincs_np(old_pos, new_pos, bond, boxl, nchain, lchain, r0, Sdiag, con_index, con_coeff, atom1, atom2, ncc, nrec=2):

	N = nchain * lchain

	K = nchain * (lchain-1)
	cmax = lchain-1
	rhs = np.zeros((2, K))
	B = np.zeros((K, 2))
	solution = np.zeros(K)

	bonds = np.where(np.triu(bond))
	nbonds = int(np.sum(bond) / 2)
	bond_check = np.transpose((bonds[0], bonds[1]))

	print(con_index, ncc)
	print(con_coeff)
	print(np.transpose((atom1, atom2)))
	print(bond_check)

	ncc = np.zeros(nbonds, dtype=int)
	bond_index = np.zeros((nbonds, nbonds))

	for i in range(nbonds):
		for j in range(i):
			bond_index[i][j] = np.any(np.in1d(bond_check[i], bond_check[j]))
			bond_index[j][i] = np.any(np.in1d(bond_check[i], bond_check[j]))

	print(bond_index)
	print(np.where(np.triu(bond_index)))

	for i in range(nbonds):
		for j in range(nbonds):
			if i != j:
				ncc[i] += np.sum(np.in1d(bond_check[i], bond_check[j]))

	atoms = np.rot90(bond_check)

	"""Form B matrix (shape = (nbonds, 2))"""
	old_dx, old_dy = get_dx_dy(old_pos, N, boxl)
	old_dx = old_dx[np.where(np.triu(bond))]
	old_dy = old_dy[np.where(np.triu(bond))]
	old_r = np.sqrt(old_dx**2 + old_dy**2)

	new_dx, new_dy = get_dx_dy(new_pos, N, boxl)
	new_dx = new_dx[np.where(np.triu(bond))]
	new_dy = new_dy[np.where(np.triu(bond))]
	new_r = np.sqrt(old_dx**2 + old_dy**2)

	B_x = old_dx / old_r
	B_y = old_dy / old_r

	B = np.transpose((B_x, B_y))
	dxy = np.transpose((new_dx, new_dy))

	print(B.shape, B)
	#print(dxy)
	#print(np.sum(B*dxy, axis=1))

	A = np.zeros((K, cmax))

	print(ncc)

	for i in range(nbonds):
		for j in range(nbonds):
			A[i,j] = con_coeff[i,j] * (B[i,0]*B[j,0] + B[i,1]*B[j,1])

	print(A)
	A = np.zeros((K, cmax))

	for i in range(K):
		a1 = atom1[i]
		a2 = atom2[i]
		for n in range(ncc[i]):
			k = con_index[i,n]
			print(i, a1, a2, n, k)
			A[i,n] = con_coeff[i,n] * (B[i,0]*B[k,0] + B[i,1]*B[k,1])  

			dxy = (new_pos[a1] - new_pos[a2])
			dxy -= boxl * np.array(2 * dxy/ boxl, dtype=int)

			print(B[i], dxy)
			print(B[i] * dxy)
			print(np.sum(B[i] * dxy))
			
			rhs[0,i] = Sdiag * (np.sum(B[i] * dxy) - r0)
			solution[i]=rhs[0,i]

	print(A)
	#print(rhs)

	sys.exit()

	new_pos = lincs_solve(new_pos, K, atom1, atom2, ncc, con_index, Sdiag, B, A, rhs, solution)

	for i in range(K):
		a1 = atom1[i]
		a2 = atom2[i]

		dxy = (new_pos[a1] - new_pos[a2])
		dxy -= boxl * np.array(2 * dxy/ boxl, dtype=int)

		p = np.sqrt(2 * r0**2 - np.sum(dxy**2))
		rhs[0,i] = Sdiag * (r0 - p)
		solution[i] = rhs[0,i]
 
	new_pos = lincs_solve(new_pos, K, atom1, atom2, ncc, con_index, Sdiag, B, A, rhs, solution)

	return new_pos


