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
	sc.set_offsets(np.c_[tot_pos[n][0], tot_pos[n][1]])
	

nsteps = 1#5000
nchain = 2
lchain = 10
N = nchain * lchain

if nchain > 1: n_section = np.sqrt(np.min([i for i in np.arange(nchain+1)**2 if i >= nchain]))
else: n_section = 1

sig1 = 2.
boxl = N * sig1
bsize = sig1 * 300
ep1 = 3.0
dt = 0.005
T = 10

r0 = 2. **(1./6.) * sig1
kB = 50.
rc = 4 * sig1

tot_pos = np.zeros((nsteps, N, 2))
tot_vel = np.zeros((nsteps, N, 2))
tot_frc = np.zeros((nsteps, N, 2))

pos, vel, frc, bond = ut.setup(boxl, nchain, lchain, T, sig1, ep1, r0, kB, rc)

energy_array = np.zeros(nsteps)

for n in range(nsteps):

	pos, vel, frc = ut.VV_alg(pos, vel, frc, bond, dt, N, boxl, sig1, ep1, r0, kB, rc)

	#energy = ut.tot_energy(N, pos, bond, boxl, sig1, ep1, r0, kB, rc)
	#energy_array[n] += energy

	sys.stdout.write("STEP {}\r".format(n))
	sys.stdout.flush()

	if np.sum(np.abs(vel)) >= 1000: 
		print("velocity exceeded, step ={}".format(n))
		break 

	tot_pos[n] += pos
	tot_vel[n] += vel
	tot_frc[n] += frc


#plt.plot(energy_array)
#plt.show()

tot_pos = np.moveaxis(tot_pos, 2, 1)

fig, ax = plt.subplots()

sc = ax.scatter(tot_pos[0][0], tot_pos[0][1])
plt.xlim(0, boxl)
plt.ylim(0, boxl)
ani = animation.FuncAnimation(fig, animate, frames=nsteps, interval=100, repeat=False)
plt.show()	





