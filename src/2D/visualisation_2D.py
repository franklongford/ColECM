"""
COLLAGEN FIBRE SIMULATION 2D VISULISATION

Created by: Frank Longford
Created on: 01/11/15

Last Modified: 21/03/2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os

import utilities_2D as ut


def make_gif(file_name, fig_dir, gif_dir, n_frame, images, res, sharp, cell_dim, itype='MD'):

	import imageio

	image_list = []
	file_name_plot = '{}_{}_{}'.format(file_name, res, sharp)

	for frame in range(n_frame):

		if not os.path.exists('{}/{}_{}_ISM.png'.format(fig_dir, file_name_plot, frame)):
			if itype.upper() == 'MD': 
				fig, ax = plt.subplots(figsize=(cell_dim[0]/4, cell_dim[1]/4))
				plt.scatter(images[frame][0], images[frame][1])
				plt.xlim(0, cell_dim[0])
				plt.ylim(0, cell_dim[1])
			elif itype.upper() == 'SHG':
				fig = plt.figure()
				plt.imshow(images[frame], cmap='viridis', interpolation='nearest', extent=[0, cell_dim[0], 0, cell_dim[1]], origin='lower')
				#plt.gca().set_xticks(np.linspace(0, cell_dim[0], 10))
				#plt.gca().set_yticks(np.linspace(0, cell_dim[1], 10))
				#plt.gca().set_xticklabels(real_x)
				#plt.gca().set_yticklabels(real_y)
			plt.savefig('{}/{}_{}_heat.png'.format(fig_dir, file_name_plot, frame), bbox_inches='tight')
			plt.close()

		image_list.append('{}/{}_{}_heat.png'.format(fig_dir, file_name_plot, frame))

	file_name_gif = '{}_{}_{}_{}'.format(file_name, res, sharp, n_frame)
	file_path_name = '{}/{}.gif'.format(gif_dir, file_name_gif)

	with imageio.get_writer(file_path_name, mode='I', duration=0.3, format='GIF') as writer:
		for filename in image_list:
			image = imageio.imread(filename)
			writer.append_data(image)
			#os.remove(filename)


def form_n_vector(dx_shg, dy_shg):

	r_xy = np.zeros(dx_shg.shape)
	tx = np.zeros(dx_shg.shape)
	ty = np.zeros(dx_shg.shape)

	r_xy_2 = (dx_shg**2 + dy_shg**2)
	indicies = np.where(r_xy_2 > 0)
	r_xy[indicies] += np.sqrt(r_xy_2[indicies].flatten())

	tx[indicies] += dx_shg[indicies] / r_xy[indicies]
	ty[indicies] -= dy_shg[indicies] / r_xy[indicies]

	nxx = tx**2
	nyy = ty**2
	nxy = tx*ty

	n_vector = np.array((nxx, nxy, nxy, nyy))
	n_vector = np.moveaxis(n_vector, (1, 3, 2, 0), (0, 1, 2, 3))

	return n_vector


def alignment_analysis(n_vector, area):

	n_frame = n_vector.shape[0]
	n_x = n_vector.shape[1]
	n_y = n_vector.shape[2]
	pixel_xy = np.array((n_x / area, n_y / area), dtype=int)
	print(n_x, n_y, pixel_xy*area)
	av_n1 = np.zeros((n_frame, area, area))
	av_n2 = np.zeros((n_frame, area, area))
	av_n_vec = np.zeros((n_frame, area, area, 2, 2))

	for x in range(area):
		for y in range(area):
			cut_n_vector = n_vector[:, pixel_xy[0] * x: pixel_xy[0] * (x+1), 
									pixel_xy[1] * y: pixel_xy[1] * (y+1), :]
			cut_n_vector = np.moveaxis(cut_n_vector, (0, 3, 1, 2), (0, 1, 2, 3))
			av_n = np.reshape(np.mean(cut_n_vector, axis=(2, 3)), (n_frame, 2, 2))

			for frame in range(n_frame):
				eig_val, eig_vec = np.linalg.eig(av_n[frame])

				av_n1[frame][y][x] += eig_val[0]
				av_n2[frame][y][x] += eig_val[1]
				av_n_vec[frame][y][x] += eig_vec

	return av_n1, av_n2, av_n_vec
	

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

if ('-gif' in sys.argv): gif_file_name = sys.argv[sys.argv.index('-gif') + 1]
else: gif_file_name = input("Enter gif_file name: ")


param_file = ut.read_param_file(param_file_name)
cell_dim = param_file['cell_dim']
vdw_param = param_file['vdw_param']
rc = param_file['rc']

tot_pos = ut.load_npy(traj_file_name)
n_frame = tot_pos.shape[0]

res = 5
sharp = 2

n_x = int(cell_dim[0] * res)
n_y = int(cell_dim[1] * res)

gif_dir = current_dir + '/gif'
if not os.path.exists(gif_dir): os.mkdir(gif_dir)
fig_dir = current_dir + '/fig'
if not os.path.exists(fig_dir): os.mkdir(fig_dir)

skip = 10
n_image = int(n_frame/skip)

tot_pos = np.moveaxis(tot_pos, 2, 1)
image_md = np.array([tot_pos[n] for n in range(0, n_frame, skip)])

image_shg, dx_shg, dy_shg = ut.shg_images(image_md, 2 * vdw_param[0] * sharp, n_x, n_y, rc * sharp)
n_vector = form_n_vector(dx_shg, dy_shg)
area = 5
n1_shg, n2_shg, n_shg = alignment_analysis(n_vector, area)

plt.figure(0)
plt.imshow(image_shg[-1], cmap='viridis', extent=[0, cell_dim[0], 0, cell_dim[1]], origin='lower')
plt.figure(1)
plt.imshow(dx_shg[-1], cmap='viridis', extent=[0, cell_dim[0], 0, cell_dim[1]], origin='lower')
plt.figure(2)
plt.imshow(abs(n1_shg - n2_shg)[-1], cmap='viridis', extent=[0, cell_dim[0], 0, cell_dim[1]], origin='lower')
plt.show()

make_gif(gif_file_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, res, sharp, cell_dim, 'SHG')
make_gif(gif_file_name + '_MD', fig_dir, gif_dir, n_image, image_md, res, sharp, cell_dim, 'MD')

"""
hist, image = ut.create_image(tot_pos[-1], 2 * vdw_param[0] * sharp, n_x, n_y)

fig = plt.figure(figsize=(cell_dim[0]/4, cell_dim[1]/4))
plt.scatter(tot_pos[-1].T[0], tot_pos[-1].T[1])
plt.axis([0, cell_dim[0], 0, cell_dim[1]])
plt.savefig('{}/{}_pos_sample.png'.format(gif_dir, gif_file_name), bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.imshow(hist, cmap='viridis', extent=[0, cell_dim[0], 0, cell_dim[1]], origin='lower')
#plt.gca().set_xticks(np.linspace(0, cell_dim[0], 10))
#plt.gca().set_yticks(np.linspace(0, cell_dim[1], 10))
plt.savefig('{}/{}_{}_hist_sample.png'.format(gif_dir, gif_file_name, res), bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.imshow(image, cmap='viridis', extent=[0, cell_dim[0], 0, cell_dim[1]], origin='lower')
#plt.gca().set_xticks(np.linspace(0, cell_dim[0], 10))
#plt.gca().set_yticks(np.linspace(0, cell_dim[1], 10))
plt.savefig('{}/{}_{}_{}_gauss_sample.png'.format(gif_dir, gif_file_name, res, sharp), bbox_inches='tight')
plt.close()

fig, ax = plt.subplots()
plt.imshow(image_pos[0], cmap='viridis', interpolation='nearest')
ani = animation.FuncAnimation(fig, heatmap_animation, frames=n_frame, interval=100, repeat=False)
plt.show()

fig, ax = plt.subplots()
sc = ax.scatter(tot_pos[0][0], tot_pos[0][1])
plt.xlim(0, cell_dim[0])
plt.ylim(0, cell_dim[1])
ani = animation.FuncAnimation(fig, animate, frames=n_frame, interval=100, repeat=False)
plt.show()
"""

