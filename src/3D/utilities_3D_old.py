import numpy as np
import os
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

""" STANDARD ROUTINES """


def unit_vector(vector):

	sum_ = np.sum([i**2 for i in vector])
	norm = 1./sum_
	return np.array([np.sqrt(i**2 * norm) * np.sign(i) for i in vector])


def rand_vector(n): return unit_vector(np.random.random((n)) * 2 - 1) 


""" Molecular Mechanics Verlocity Verlet Integration """

def setup(boxl, nchain, lchain, sig1, ep1, r0, kB):

	N = nchain*lchain
	pos = np.zeros((N, 3))
	bond = np.zeros((N, N))

	for chain in xrange(nchain):
		for bead in xrange(lchain):
			i = chain * nchain + bead
			pos, bond[i][i-1], bond[i-1][i] = grow_chain(bead, i, N, pos, sig1, ep1, r0, kB, bond, boxl, 1E6)

	vel = (np.random.random((N,3)) - 0.5) * 2
	frc = calc_forces(N, boxl, pos, bond, sig1, ep1, r0, kB)

	return pos, vel, frc, bond


def grow_chain(bead, i, N, pos, sig1, ep1, r0, kB, bond, boxl, max_energy):

	if bead == 0:
		pos[i] = np.random.random((3)) * boxl  
		return  pos, 0, 0
	else:
		energy = max_energy +1
		while  energy > max_energy:
			pos[i] = pos[i-1] + rand_vector(3) * sig1
			energy = tot_energy(N, pos, bond, boxl, sig1, ep1, r0, kB)
			
		for n in xrange(3): pos[i][n] += boxl * (1 - int((pos[i][n] + boxl) / boxl))
		return pos, 1, 1


def calc_forces(N, boxl, pos, bond, sig1, ep1, r0, kB):

	f_beads = np.zeros((N, 3))

	for i in xrange(N):
		for j in xrange(i):
			dx = (pos[i][0] - pos[j][0])
			dx -= boxl * int(2*dx/boxl)
			dy = (pos[i][1] - pos[j][1])
			dy -= boxl * int(2*dy/boxl)
			dz = (pos[i][2] - pos[j][2])
			dz -= boxl * int(2*dz/boxl)
			r2 = dx**2 + dy**2 + dz**2

			if bond[i][j] == 1:
				r = np.sqrt(r2)
				Fr = force_bond(r, r0, kB)
				f_beads[i][0] += dx / r * Fr
				f_beads[i][1] += dy / r * Fr
				f_beads[i][2] += dz / r * Fr

				f_beads[j][0] -= dx / r * Fr
				f_beads[j][1] -= dy / r * Fr
				f_beads[j][2] -= dz / r * Fr

			else:
				Fr = force_vdw(r2, sig1, ep1)
				f_beads[i][0] += dx / r2 * Fr
				f_beads[i][1] += dy / r2 * Fr
				f_beads[i][2] += dz / r2 * Fr

				f_beads[j][0] -= dx / r2 * Fr
				f_beads[j][1] -= dy / r2 * Fr
				f_beads[j][2] -= dz / r2 * Fr

			#print "{} {} {}".format(x, y, r)

	return f_beads


def VV_alg(pos, vel, frc, bond, dt, N, boxl, sig1, ep1, r0, kB):

	for i in xrange(N):
		for j in xrange(3):  
			vel[i][j] += 0.5 * dt * frc[i][j]
			pos[i][j] += dt * vel[i][j]
			pos[i][j] += boxl * (1 - int((pos[i][j] + boxl) / boxl))

	frc = calc_forces(N, boxl, pos, bond, sig1, ep1, r0, kB)

	for i in xrange(N): 
		for j in xrange(3): vel[i][j] += 0.5 * dt * frc[i][j]

	return pos, vel, frc


def force_bond(r, r0, kB): return 2 * kB * (r0 - r)


def force_vdw(r2, sig1, ep1): return 24 * ep1 * (2 * (sig1/r2)**6 - (sig1/r2)**3)


def pot_vdw(r2, sig1, ep1): return 4 * ep1 * ((sig1**12/r2**6) - (sig1**6/r2**3))


def pot_bond(r, r0, kB): return kB * (r - r0)**2


def tot_energy(N, pos, bond, boxl, sig1, ep1, r0, kB):

	energy = 0	
	for i in xrange(N):
		for j in xrange(i):
			if np.dot(pos[i], pos[j]) != 0:
				dx = (pos[i][0] - pos[j][0])
				dx -= boxl * int(2*dx/boxl)
				dy = (pos[i][1] - pos[j][1])
				dy -= boxl * int(2*dy/boxl)
				dz = (pos[i][2] - pos[j][2])
				dz -= boxl * int(2*dz/boxl)
				r2 = dx**2 + dy**2 + dz**2

				if bond[i][j] == 1:
					r = np.sqrt(r2)
					energy += pot_bond(r, r0, kB)

				else: energy += pot_vdw(r2, sig1, ep1)
	return energy 


""" Dissipative Particle Dynamics Verlocity Verlet Integration """

def setup_DPD(boxl, nchain, lchain, a, gamma, sigma, sqrt_dt, r_m, K):

	N = nchain * lchain
	chains = np.zeros(N)
	beads = np.zeros(N)
	polar = np.zeros(N)
	bond = np.zeros((N, N))
	A = np.zeros((N,N))
	p = 1

	pos = np.zeros((N, 3))

	for i in xrange(N): 
		chains[i] = int(i / lchain)
		beads[i] = i % lchain
		pos, bond[i][i-1], bond[i-1][i] = grow_chain_DPD(i, beads[i], pos, sigma, boxl)
		if np.random.rand() < 0.2: p = -p
		polar[i] = p 

	for i in xrange(N): 
		for j in xrange(i):
			if chains[i] == chains[j]: A[i][j] = a[0]
			else: A[i][j] = a[1]

	vel = (np.random.random((N, 3)) - 0.5) * 2
	frc = calc_forces_DPD(N, boxl, pos, vel, bond, polar, A, gamma, sigma, sqrt_dt, r_m, K, chains, beads)

	return pos, vel, frc, bond, polar, A, chains, beads


def grow_chain_DPD(i, bead, pos, r_m, boxl):

	if bead == 0:
		pos[i] = np.random.random((3)) * boxl  
		return  pos, 0, 0
	else:
		pos[i] = pos[i-1] + rand_vector(3) * r_m	
		for n in xrange(3): pos[i][n] += boxl * (1 - int((pos[i][n] + boxl) / boxl))
		return pos, 1, 1


def make_bond_DPD(i, j, polar, bond, chains, beads):

	chaini = int(i / lchain)
	chainj = int(j / lchain)
	beadi = i % lchain
	beadj = j % lchain

	if beadi == 0 or (i+1) % lchain == lchain-1:
		if beadj == 0 or beadj == lchain-1:
			if polar[i] == polar[j] and np.random.rand() < 0.001: 
				bond[i][j] = 1
				bond[j][i] = 1
				print "Bond formed, N = {} {}, TYPE = {} {}".format(i, j, TYPE[i], TYPE[j])
				for n in xrange(chaini*lchain, (chaini+1)*lchain): TYPE[n] = np.min([TYPE[i],TYPE[j]])
				for n in xrange(chainj*lchain, (chainj+1)*lchain): TYPE[n] = np.min([TYPE[i],TYPE[j]])
				start = int(np.min([chaini,chainj])*lchain)
				end = int(np.min([chaini,chainj])*lchain + lchain)
				for n in xrange(start, 	end):

	return bond, chains, beads


def calc_forces_DPD(N, boxl, pos, vel, bond, polar, A, gamma, sigma, sqrt_dt, r_m, K, chains, beads):
			
	f_beads = np.zeros((N, 3))

	for i in xrange(N):
		for j in xrange(i):
	
			dx = (pos[i][0] - pos[j][0])
			dx -= boxl * int(2*dx/boxl)
			dy = (pos[i][1] - pos[j][1])
			dy -= boxl * int(2*dy/boxl)
			dz = (pos[i][2] - pos[j][2])
			dz -= boxl * int(2*dz/boxl)
			r2 = dx**2 + dy**2 + dz**2	
			
			if r2 < r_m**2 and bond[i][j] == 0: make_bond_DPD(i, j, polar, bond, chains, beads)
			if bond[i][j] == 1:
				
				r = np.sqrt(r2)
				r_uvector = unit_vector([dx, dy, dz])
				a = A[i][j]

				Fr = DPD_force_C(r, a, r_uvector)

				f_beads[i] += Fr
				f_beads[j] -= Fr			

			elif r2 < 1:
				r = np.sqrt(r2)
				a = A[i][j]
				r_uvector = unit_vector([dx, dy, dz])
				v_vector = [(vel[i][0] - vel[j][0]), (vel[i][1] - vel[j][1]), (vel[i][2] - vel[j][2])]

				Fr = DPD_force_C(r, a, r_uvector)
				Fr += DPD_force_D(r, gamma, r_uvector, v_vector)		
				Fr += DPD_force_R(r, sigma, sqrt_dt, r_uvector)

				f_beads[i] += Fr
				f_beads[j] -= Fr

	return f_beads


def VV_alg_DPD(pos, vel, frc, dt, N, boxl, bond, polar, A, gamma, sigma, lam, sqrt_dt, r_m, K, chains, beads):

	new_vel = np.zeros((N, 3))

	for i in xrange(N):
		for j in xrange(3):
			new_vel[i][j] += lam * dt * frc[i][j]
			pos[i][j] += dt * vel[i][j] + 0.5 * dt**2 * frc[i][j]
			pos[i][j] += boxl * (1 - int((pos[i][j] + boxl) / boxl))

	new_frc = calc_forces_DPD(N, boxl, pos, new_vel, bond, polar, A, gamma, sigma, sqrt_dt, r_m, K, chains, beads)
			
	for i in xrange(N): 
		for j in xrange(3): vel[i][j] += 0.5 * dt * (frc[i][j] + new_frc[i][j])

	return pos, vel, new_frc, chains, beads


def DPD_force_C(r, a, r_uvector): return a * omegaR(r) * r_uvector


def DPD_force_D(r, gamma, r_uvector, v_vector): return - gamma * omegaR(r)**2 * r * np.dot(r_uvector, v_vector) * r_uvector


def DPD_force_R(r, sigma, sqrt_dt, r_uvector): return sigma * omegaR(r) * (np.random.rand()*2 - 1) * r_uvector / sqrt_dt


def DPD_force_FENE(r, r_m, K, r_uvector): return K * r_uvector / (1 - (r / r_m))


def omegaR(r): return 1. - r


def save_system(root, pos, vel, frc, bond, TYPE, nchain, lchain):

	if not os.path.exists(root): os.mkdir(root)

	with file('{}/{}_{}_POS.npz'.format(root, nchain, lchain), 'w') as outfile:
		np.savez(outfile, pos=pos, fmt='%-12.6f')
	with file('{}/{}_{}_VEL.npz'.format(root, nchain, lchain), 'w') as outfile:
		np.savez(outfile, vel=vel, fmt='%-12.6f')
	with file('{}/{}_{}_FRC.npz'.format(root, nchain, lchain), 'w') as outfile:
		np.savez(outfile, frc=frc, fmt='%-12.6f')
	with file('{}/{}_{}_BOND.npz'.format(root, nchain, lchain), 'w') as outfile:
		np.savez(outfile, bond=bond)
	with file('{}/{}_{}_TYPE.npz'.format(root, nchain, lchain), 'w') as outfile:
		np.savez(outfile, TYPE=TYPE)

def load_system(root, nchain, lchain):

	with file('{}/{}_{}_POS.npz'.format(root, nchain, lchain), 'r') as infile:
		npzfile = np.load(infile)
		pos = npzfile['pos']
	with file('{}/{}_{}_VEL.npz'.format(root, nchain, lchain), 'r') as infile:
		npzfile = np.load(infile)
		vel = npzfile['vel']
	with file('{}/{}_{}_FRC.npz'.format(root, nchain, lchain), 'r') as infile:
		npzfile = np.load(infile)
		frc = npzfile['frc']
	with file('{}/{}_{}_BOND.npz'.format(root, nchain, lchain), 'r') as infile:
		npzfile = np.load(infile)
		bond = npzfile['bond']
	with file('{}/{}_{}_TYPE.npz'.format(root, nchain, lchain), 'r') as infile:
		npzfile = np.load(infile)
		TYPE = npzfile['TYPE']

	return pos, vel, frc, bond, TYPE


""" Visualisation of System """

def plot_system(pos, vel, frc, N, L, bsize):

	width = 0.2
	plt.ion()
	positions = np.rot90(np.array(pos))
	fig = plt.figure(0, figsize=(15,15))
	fig.clf()

	ax = plt.subplot(2,1,1)
	ax = fig.gca(projection='3d')
	ax.scatter(positions[0], positions[1], positions[2], c=range(N), s=bsize)
	ax.set_xlim3d(0, L)
	ax.set_ylim3d(0, L)
	ax.set_zlim3d(0, L)
	fig.canvas.draw()

	velocities = np.rot90(vel)
	forces = np.rot90(frc)

	fig = plt.figure(1, figsize=(15,15))
	fig.clf()

	ax = plt.subplot(2,1,1)
	ax.set_ylim(-10,10)
	vel_x = ax.bar(range(N), velocities[0], width, color='b')
	vel_y = ax.bar(np.arange(N)+width, velocities[1], width, color='g')
	vel_z = ax.bar(np.arange(N)+2*width, velocities[2], width, color='r')

	ax = plt.subplot(2,1,2)
	ax.set_ylim(-20,20)
	frc_x = ax.bar(range(N), forces[0], width, color='b')
	frc_y = ax.bar(np.arange(N)+width, forces[1], width, color='g')
	frc_z = ax.bar(np.arange(N)+2*width, forces[2], width, color='r')
	fig.canvas.draw()


def plot_system_DPD(pos, vel, frc, bond, N, L, bsize, TYPE, n):

	width = 0.2
	plt.ion()

	positions = np.rot90(pos)
	velocities = np.rot90(vel)
	forces = np.rot90(frc)

	fig = plt.figure(0, figsize=(15,15))
	plt.title(n)
	fig.clf()
	
	ax = plt.subplot(2,1,1)
	ax.set_ylim(-10,10)
	vel_x = ax.bar(range(N), velocities[0], width, color='b')
	vel_y = ax.bar(np.arange(N)+width, velocities[1], width, color='g')
	vel_z = ax.bar(np.arange(N)+2*width, velocities[2], width, color='r')

	ax = plt.subplot(2,1,2)
	ax.set_ylim(-20,20)
	frc_x = ax.bar(range(N), forces[0], width, color='b')
	frc_y = ax.bar(np.arange(N)+width, forces[1], width, color='g')
	frc_z = ax.bar(np.arange(N)+2*width, forces[2], width, color='r')
	fig.canvas.draw()

	fig = plt.figure(1, figsize=(15,15))
	plt.title(n)
	fig.clf()

	COLOUR = ['b', 'g', 'r']

	ax = plt.subplot(2,1,1)
	ax = fig.gca(projection='3d')
	ax.scatter(positions[0], positions[1], positions[2], c=TYPE, s=bsize)
	#for i in xrange(N):	
	#	for j in xrange(i):
	#		if bond[i][j] == 1: ax.plot([positions[0][i], positions[0][j]], [positions[1][i], positions[1][j]], 
	#		[positions[2][i], positions[2][j]], c=COLOUR[int(TYPE[i])])
	ax.set_xlim3d(0, L)
	ax.set_ylim3d(0, L)
	ax.set_zlim3d(0, L)
	fig.canvas.draw()

	

