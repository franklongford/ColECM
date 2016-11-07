import numpy as np
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation


def unit_vector(vector):

	sum_ = np.sum([i**2 for i in vector])
	norm = 1./sum_
	return [np.sqrt(i**2 * norm) * np.sign(i) for i in vector]


def rand_vector(n): return np.array(unit_vector(np.random.random((n)) * 2 - 1)) 


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

				f_beads[j][0] -= dx / r2 * Fr
				f_beads[j][1] -= dy / r2 * Fr
				f_beads[j][2] -= dz / r2 * Fr

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


def calc_forces_DPD(N, boxl, pos, vel, a, gamma, sigma):

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
			r = np.sqrt(r2)

			v_vector = [(vel[i][0] - vel[j][0]), (vel[i][1] - vel[j][1]), (vel[i][2] - vel[j][2])]

			r_uvector(unit_vector([dx, dy, dz]))

			Fr = DPD_force_C(r, a, r_uvector)
			Fr += DPD_force_D(r, gamma, r_uvector, v_vector)		
			Fr += DPD_force_R(r, sigma, theta, r_uvector)

			f_beads[i][0] += dx / r * Fr
			f_beads[i][1] += dy / r * Fr
			f_beads[i][2] += dz / r * Fr

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


def VV_alg_DPD(pos, vel, frc, dt, N, boxl, a, gamma, sigma, lam):

	new_vel = np.zeros((N, 3))

	for i in xrange(N):
		for j in xrange(3):
			new_vel[i][j] += lam * dt * frc[i][j]
			pos[i][j] += dt * vel[i][j] + 0.5 * dt**2 * frc[i][j]
			pos[i][j] += boxl * (1 - int((pos[i][j] + boxl) / boxl))

	new_frc = calc_forces_DPD(N, boxl, pos, vel, a, gamma, sigma)
			
	for i in xrange(N): 
		for j in xrange(3): vel[i][j] += 0.5 * dt * (frc[i][j] + new_frc[i][j])

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


def plot_system(pos, vel, frc, N, L, bsize):

	width = 0.2
	plt.ion()
	positions = np.rot90(pos)
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


def DPD_force_C(r, a, r_uvector):
	if r < 1: return a * omegaR(r) * r_uvector
	else: return 0


def DPD_force_D(r, gamma, r_uvector, v_vector):
	if r < 1: return - gamma * omegaR(r)**2 * r * np.dot(r_uvector, v_vector) * r_uvector
	else: return 0

def DPD_force_R(r, sigma, theta, r_uvector):
	if r < 1: return sigma * omegaR(r) * r * theta * r_uvector
	else: return 0

def omegaR(r): return 1 - r

