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
import matplotlib.animation as animation

import sys
import os

import utilities_2D as ut

def animate(n):
	plt.title('Frame {}'.format(n * speed))
	sc.set_offsets(np.c_[tot_pos[n][0], tot_pos[n][1]])


def cum_mov_average(array):

	l = len(array)
	average = np.zeros(l)
	average[0] = array[0]

	for i in range(l-1):
		average[i+1] = average[i] + (array[i+1] - average[i]) / (i+1)  
	
	return average

nsteps = 10000
nchain = 1
lchain = 5
N = nchain * lchain

sig1 = 1.
boxl = nchain * sig1 * lchain**2 * 0.5
print(boxl)
bsize = sig1 * 300
ep1 = 5.0
dt = 0.002
T = 1.

r0 = 2. **(1./6.) * sig1
kB = 100.
rc = 4 * sig1

m = 1.
Sdiag = np.sqrt(m / 2.)

tot_pos = np.zeros((nsteps, N, 2))
tot_vel = np.zeros((nsteps, N, 2))
tot_frc = np.zeros((nsteps, N, 2))

pos, vel, frc, bond, boxl, con_index, con_coeff, atom1, atom2, ncc = ut.setup(boxl, nchain, lchain, T, sig1, ep1, r0, kB, rc, Sdiag)
print(boxl)

dx, dy = ut.get_dx_dy(pos, N, boxl)
r2 = dx**2 + dy**2
verlet_list = ut.check_cutoff(r2, rc**2)

energy_array = np.zeros(nsteps)

print('\n')

for n in range(nsteps):

	#pos, vel, frc, verlet_list, energy = ut.VV_alg(n, pos, vel, frc, bond, verlet_list, dt, N, boxl, sig1, ep1, r0, kB, rc)

	pos, vel, frc, verlet_list, energy = ut.VV_alg(n, pos, vel, frc, bond, verlet_list, dt, nchain, lchain, atom1, atom2, 
													con_index, con_coeff, ncc, boxl, sig1, ep1, r0, kB, rc, Sdiag)

	energy_array[n] += energy

	sys.stdout.write("STEP {}\r".format(n))
	sys.stdout.flush()

	if np.sum(np.abs(vel)) >= T * 1000: 
		print("velocity exceeded, step ={}".format(n))
		break 

	tot_pos[n] += pos
	tot_vel[n] += vel
	tot_frc[n] += frc

CMA = cum_mov_average(energy_array) / N
plt.plot(CMA)
plt.show()

speed = 10

tot_pos = np.array([tot_pos[i] for i in range(nsteps) if i % speed == 0])

tot_pos = np.moveaxis(tot_pos, 2, 1)

fig, ax = plt.subplots()

sc = ax.scatter(tot_pos[0][0], tot_pos[0][1])
plt.xlim(0, boxl)
plt.ylim(0, boxl)
ani = animation.FuncAnimation(fig, animate, frames=int(nsteps/speed), interval=100, repeat=False)
plt.show()	





