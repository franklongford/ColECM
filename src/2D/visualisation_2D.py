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

current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv) < 2: traj_file_name = current_dir + '/' + input("Enter traj_file name: ")
else: traj_file_name = current_dir + '/' + sys.argv[1]
if len(sys.argv) < 3: param_file_name = current_dir + '/' + input("Enter param_file name: ")
else: param_file_name = current_dir + '/' + sys.argv[2]


param_file = ut.read_param_file(param_file_name)
cell_dim = param_file['cell_dim']

tot_pos = ut.load_npy(traj_file_name)

tot_pos = np.moveaxis(tot_pos, 2, 1)

fig, ax = plt.subplots()

sc = ax.scatter(tot_pos[0][0], tot_pos[0][1])
plt.xlim(0, cell_dim[0])
plt.ylim(0, cell_dim[1])
ani = animation.FuncAnimation(fig, animate, frames=tot_pos.shape[0], interval=100, repeat=False)
plt.show()	

