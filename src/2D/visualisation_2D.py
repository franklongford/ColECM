"""
COLLAGEN FIBRE SIMULATION 2D VISULISATION

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
	plt.title('Frame {}'.format(n))
	sc.set_offsets(np.c_[tot_pos[n][0], tot_pos[n][1]])


if len(sys.argv) < 2: directory = raw_input("Enter traj file: ")
else: traj_file_name = sys.argv[1]

if len(sys.argv) < 3: directory = raw_input("Enter restart file: ")
else: restart_file_name = sys.argv[2]


pos = np.load(restart_file_name)
cell_dim = pos[-1]
pos = pos[:-1]

tot_pos = np.load(traj_file_name)

tot_pos = np.moveaxis(tot_pos, 2, 1)

fig, ax = plt.subplots()

sc = ax.scatter(tot_pos[0][0], tot_pos[0][1])
plt.xlim(0, cell_dim[0])
plt.ylim(0, cell_dim[1])
ani = animation.FuncAnimation(fig, animate, frames=tot_pos.shape[0], interval=100, repeat=False)
plt.show()	

