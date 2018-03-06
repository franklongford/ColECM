"""
COLLAGEN FIBRE SIMULATION 2D

Created by: Frank Longford
Created on: 01/11/15

Last Modified: 06/03/2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os

import utilities_2D as ut

def animate(n):
	plt.title('Frame {}'.format(n * speed))
	sc.set_offsets(np.c_[tot_pos[n][0], tot_pos[n][1]])


n_steps = 2000
n_fibre = 10
l_fibre = 5
N = n_fibre * l_fibre

mass = 1.

dt = 0.002

vdw_sigma = 1.
vdw_epsilon = 5.0
vdw_param = [vdw_sigma, vdw_epsilon]

bond_r0 = 2. **(1./6.) * vdw_sigma
bond_k = 20.
bond_param = [bond_r0, bond_k]

angle_theta0 = np.pi
angle_k = 20.
angle_param = [angle_theta0, angle_k]

rc = 4 * vdw_sigma

cell_L = n_fibre * vdw_sigma**2 * l_fibre
cell_dim = np.array([cell_L, cell_L], dtype=float)
tile_cell_dim = np.tile(cell_dim, (N, 1))

tot_pos = np.zeros((n_steps, N, 2))
tot_vel = np.zeros((n_steps, N, 2))
tot_frc = np.zeros((n_steps, N, 2))

Langevin = True
kBT = 5.
thermo_gamma = 2.0
thermo_sigma =  np.sqrt(2 * kBT * thermo_gamma / mass)
thermo_xi = np.random.normal(0, 1, (n_steps, N, 2))
thermo_theta = np.random.normal(0, 1, (n_steps, N, 2))

energy_array = np.zeros(n_steps)

pos, vel, frc, cell_dim, bond_matrix, verlet_list, atoms, dxdy_index, r_index = ut.setup_test(cell_dim, n_fibre, l_fibre, mass, kBT, vdw_param, bond_param, angle_param, rc)

print(cell_dim)

dx, dy = ut.get_dx_dy(pos, N, cell_dim)
r2 = dx**2 + dy**2
verlet_list = ut.check_cutoff(r2, rc**2)

print('\n')

for step in range(n_steps):

	pos, vel, frc, verlet_list, energy = ut.velocity_verlet_alg(pos, vel, frc, mass, bond_matrix, verlet_list, atoms, dxdy_index, r_index, 
						dt, cell_dim, vdw_param, bond_param, angle_param, rc, kBT, thermo_gamma, 
						thermo_sigma, thermo_xi[step], thermo_theta[step], Langevin)

	energy_array[step] += energy

	sys.stdout.write("STEP {}\r".format(step))
	sys.stdout.flush()

	if np.sum(np.abs(vel)) >= kBT * 1E5: 
		print("velocity exceeded, step ={}".format(step))
		print(pos)
		print(vel)
		n_steps = step
		break 

	tot_pos[step] += pos
	tot_vel[step] += vel
	tot_frc[step] += frc



CMA = ut.cum_mov_average(energy_array[:n_steps]) / N
plt.plot(CMA)
plt.show()

speed = 100

tot_pos = np.array([tot_pos[i] for i in range(n_steps) if i % speed == 0])

tot_pos = np.moveaxis(tot_pos, 2, 1)

fig, ax = plt.subplots()

sc = ax.scatter(tot_pos[0][0], tot_pos[0][1])
plt.xlim(0, cell_dim[0])
plt.ylim(0, cell_dim[1])
ani = animation.FuncAnimation(fig, animate, frames=int(n_steps/speed), interval=100, repeat=False)
plt.show()	





