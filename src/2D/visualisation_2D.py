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


def heatmap_animation(n):
	plt.title('Frame {}'.format(n))
	ax.pcolor(image_pos[n], cmap='viridis')


current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

if ('-param' in sys.argv): param_file_name = current_dir + '/' + sys.argv[sys.argv.index('-param') + 1]
else: param_file_name = current_dir + '/' + input("Enter param_file name: ")

if ('-traj' in sys.argv): traj_file_name = current_dir + '/' + sys.argv[sys.argv.index('-traj') + 1]
else: traj_file_name = current_dir + '/' + input("Enter traj_file name: ")


param_file = ut.read_param_file(param_file_name)
cell_dim = param_file['cell_dim']
vdw_param = param_file['vdw_param']

tot_pos = ut.load_npy(traj_file_name)
n_frame = tot_pos.shape[0]

size = 2
res = int(1.5 * np.sqrt(np.sum(cell_dim**2)))

for num in [0, -1]:
	image = ut.create_image(tot_pos[num], 2 * vdw_param[0] * size, size*res)
	plt.imshow(image, cmap='viridis', interpolation='nearest')
	plt.show()

"""
image_pos = np.zeros((n_frame, 75*res, 75*res))

for image in range(n_frame):
	image_pos[image] += ut.create_image(tot_pos[image], vdw_param[0]*res, 75*res)

fig, ax = plt.subplots()
#plt.imshow(image, cmap='viridis', interpolation='nearest')
ani = animation.FuncAnimation(fig, heatmap_animation, frames=n_frame, interval=100, repeat=False)
plt.show()
"""

tot_pos = np.moveaxis(tot_pos, 2, 1)

fig, ax = plt.subplots()
sc = ax.scatter(tot_pos[0][0], tot_pos[0][1])
plt.xlim(0, cell_dim[0])
plt.ylim(0, cell_dim[1])
ani = animation.FuncAnimation(fig, animate, frames=n_frame, interval=100, repeat=False)
plt.show()	

