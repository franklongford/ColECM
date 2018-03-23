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


def make_png(file_name, fig_dir, gif_dir, image, res, sharp, cell_dim, itype='MD'):

	if itype.upper() == 'MD': 
		fig, ax = plt.subplots(figsize=(cell_dim[0]/4, cell_dim[1]/4))
		plt.scatter(image[0], image[1])
		plt.xlim(0, cell_dim[0])
		plt.ylim(0, cell_dim[1])
	elif itype.upper() == 'SHG':
		fig = plt.figure()
		plt.imshow(image, cmap='viridis', interpolation='nearest', extent=[0, cell_dim[0], 0, cell_dim[1]], origin='lower')
		#plt.gca().set_xticks(np.linspace(0, cell_dim[0], 10))
		#plt.gca().set_yticks(np.linspace(0, cell_dim[1], 10))
		#plt.gca().set_xticklabels(real_x)
		#plt.gca().set_yticklabels(real_y)
	plt.savefig('{}/{}_ISM.png'.format(fig_dir, file_name), bbox_inches='tight')
	plt.close()


def make_gif(file_name, fig_dir, gif_dir, n_frame, images, res, sharp, cell_dim, itype='MD'):

	import imageio

	image_list = []
	file_name_plot = '{}_{}_{}'.format(file_name, res, sharp)

	for frame in range(n_frame):
		#if not os.path.exists('{}/{}_{}_heat.png'.format(fig_dir, file_name_plot, frame)):
		make_png("{}_{}".format(file_name_plot, frame), fig_dir, gif_dir, images[frame], res, sharp, cell_dim, itype)
		image_list.append('{}/{}_{}_ISM.png'.format(fig_dir, file_name_plot, frame))

	file_name_gif = '{}_{}_{}_{}'.format(file_name, res, sharp, n_frame)
	file_path_name = '{}/{}.gif'.format(gif_dir, file_name_gif)

	with imageio.get_writer(file_path_name, mode='I', duration=0.3, format='GIF') as writer:
		for filename in image_list:
			image = imageio.imread(filename)
			writer.append_data(image)
			#os.remove(filename)


def form_n_vector(dx_shg, dy_shg):
	"""
	form_n_vector(dx_shg, dy_shg)

	Create local nematic tensor n for each pixel in dx_shg, dy_shg

	Parameters
	----------

	dx_grid:  array_like (float); shape=(n_frame, n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(n_frame, n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	Returns
	-------

	n_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)	

	"""


	r_xy = np.zeros(dx_shg.shape)
	tx = np.zeros(dx_shg.shape)
	ty = np.zeros(dx_shg.shape)

	r_xy_2 = (dx_shg**2 + dy_shg**2)
	indicies = np.where(r_xy_2 > 0)
	r_xy[indicies] += np.sqrt(r_xy_2[indicies].flatten())

	ty[indicies] -= dx_shg[indicies] / r_xy[indicies]
	tx[indicies] += dy_shg[indicies] / r_xy[indicies]

	nxx = tx**2
	nyy = ty**2
	nxy = tx*ty

	n_vector = np.array((nxx, nxy, nxy, nyy))
	n_vector = np.moveaxis(n_vector, (1, 0, 2, 3), (0, 1, 2, 3))

	return n_vector


def alignment_analysis(n_vector, area, n_sample):
	"""
	alignment_analysis(n_vector, area, n_sample)

	Calculates eigenvalues and eigenvectors of average nematic tensor over area^2 pixels for n_samples

	Parameters
	----------

	n_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	av_eigval:  array_like (float); shape=(n_frame, n_sample, 2)
		Eigenvalues of average nematic tensors for n_sample areas

	av_eigvec:  array_like (float); shape=(n_frame, n_sample, 2, 2)
		Eigenvectors of average nematic tensors for n_sample areas

	"""

	n_frame = n_vector.shape[0]
	n_y = n_vector.shape[2]
	n_x = n_vector.shape[3]

	av_eigval = np.zeros((n_frame, n_sample, 2))
	av_eigvec = np.zeros((n_frame, n_sample, 2, 2))

	pad = int(area/2)

	for n in range(n_sample):
		start_x = np.random.randint(pad, n_x - pad)
		start_y = np.random.randint(pad, n_y - pad) 

		cut_image = image_shg[0, start_y-pad: start_y+pad, 
					 start_x-pad: start_x+pad]
		
		cut_n_vector = n_vector[:, :, start_y-pad: start_y+pad, 
					      start_x-pad: start_x+pad]

		av_n = np.reshape(np.mean(cut_n_vector, axis=(2, 3)), (n_frame, 2, 2))

		for frame in range(n_frame):
			eig_val, eig_vec = np.linalg.eigh(av_n[frame])

			av_eigval[frame][n] += eig_val
			av_eigvec[frame][n] += eig_vec

	return av_eigval, av_eigvec
	

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
l_conv = param_file['l_conv']

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
sample_l = 50
n_sample = 20
area = int(np.min([sample_l, cell_dim[0], cell_dim[1]]) / l_conv * res)

tot_pos = np.moveaxis(tot_pos, 2, 1)

image_md = np.array([tot_pos[n] for n in range(0, n_frame, skip)])

"""
test_dist = np.argwhere(np.eye(n_x, M=n_y, k=-1) + np.eye(n_x, M=n_y) + np.eye(n_x, M=n_y, k=1))
test_image, test_dx, test_dy = ut.shg_images(np.array((test_dist.T, test_dist.T)), 2 * vdw_param[0] * sharp, n_x, n_y, rc * sharp)

plt.figure(0)
plt.imshow(test_image[0], cmap='Reds', extent=[0, n_x, 0, n_y], origin='lower')
plt.figure(1)
plt.imshow(test_dx[0], cmap='coolwarm', extent=[0, n_x, 0, n_y], origin='lower')
plt.figure(2)
plt.imshow(test_dy[0], cmap='coolwarm', extent=[0, n_x, 0, n_y], origin='lower')
plt.show()

test_n_vector = form_n_vector(test_dx, test_dy)
test_eigval, test_eigvec = alignment_analysis(test_n_vector, area)
"""

"Generate Gaussian convoluted images and intensity derivatives"
image_shg, dx_shg, dy_shg = ut.shg_images(image_md, 2 * vdw_param[0] * sharp, n_x, n_y, rc * sharp)
"Calculate intensity orientational vector n for each pixel"
n_vector = form_n_vector(dx_shg, dy_shg)
"Sample average orientational anisotopy"
eigval_shg, eigvec_shg = alignment_analysis(n_vector, area, n_sample)

q = np.moveaxis(eigval_shg, (2, 0, 1), (0, 1, 2))
q = q[1] - q[0]

print('Mean anistoropy = {}'.format(np.mean(q)))

make_gif(gif_file_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, res, sharp, cell_dim, 'SHG')
make_gif(gif_file_name + '_MD', fig_dir, gif_dir, n_image, image_md, res, sharp, cell_dim, 'MD')

"""
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

