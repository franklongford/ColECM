"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 12/04/2018
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys
import os

import utilities as ut

SQRT2 = np.sqrt(2)
SQRTPI = np.sqrt(np.pi)


def reorder_array(array):
	"""
	reorder_array(array)

	Inverts 3D array so that outer loop is along z axis
	"""

	return np.moveaxis(array, (2, 0, 1), (0, 1, 2))


def move_array_centre(array, centre):
	"""
	move_array_centre(array, centre)

	Move top left corner of ND array to centre index
	"""

	n_dim = centre.shape[0]

	for i, ax in enumerate(range(n_dim)): array = np.roll(array, centre[i], axis=ax)

	return array


def gaussian(x, mean, std):
	"""
	Return value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)


def dx_gaussian(x, mean, std):
	"""
	Return derivative of value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return (mean - x) / std**2 * gaussian(x, mean, std)


def shg_images(traj, sigma, n_xyz, cut):
	"""
	shg_images(traj, sigma, n_x, n_y)

	Form a set of imitation SHG images from Gaussian convolution of a set of trajectories

	Parameters
	----------

	traj:  array_like (float); shape=(n_images, n_dim, n_bead)
		Array of sampled configurations from a simulation

	sigma:  float
		Parameter to determine variance of Guassian distribution

	n_xyz:  tuple (int); shape(n_dim)
		Number of pixels in each image dimension

	cut:  float
		Cutoff radius for convolution

	Returns
	-------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration
	"""

	n_image = traj.shape[0]
	n_dim = traj.shape[1]

	image_shg = np.zeros((n_image,) + n_xyz[:2][::-1])
	dx_shg = np.zeros((n_image,) + n_xyz[:2][::-1])
	dy_shg = np.zeros((n_image,) + n_xyz[:2][::-1])

	"Calculate distances between grid points"
	if n_dim == 2: dxdydz = np.mgrid[0:n_xyz[0], 0:n_xyz[1]]
	elif n_dim == 3: dxdydz = np.mgrid[0:n_xyz[0], 0:n_xyz[1], 0:n_xyz[2]]

	"Enforce periodic boundaries"
	for i in range(n_dim): dxdydz[i] -= n_xyz[i] * np.array(2 * dxdydz[i] / n_xyz[i], dtype=int)

	"Calculate radial distances"
	r2 = np.sum(dxdydz**2, axis=0)
	
	"Find indicies within cutoff radius"
	cutoff = np.where(r2 <= cut**2)
	"""
	"Form a filter for cutoff radius"
	filter_ = np.zeros(n_xyz)
	filter_[cutoff] += 1
	"""
	"Get all non-zero radii"
	non_zero = np.zeros(n_xyz)
	non_zero[cutoff] += 1
	non_zero[0][0] = 0

	"Form a matrix of radial distances corresponding to filter"
	r = np.sqrt(r2)
	r_cut = np.zeros(n_xyz)
	r_cut[cutoff] += np.sqrt(r2[cutoff])
	

	for i, image in enumerate(range(n_image)):
		sys.stdout.write("Processing image {} out of {}\r".format(i, n_image))
		sys.stdout.flush()

		hist, image_shg[i] = create_image(traj[image], sigma, n_xyz, r)
		dx_shg[i], dy_shg[i] = fibre_align(hist, sigma, n_xyz, dxdydz, r_cut, non_zero)

	return image_shg, dx_shg, dy_shg


def create_image(pos, std, n_xyz, r):
	"""
	create_image(pos_x, pos_y, sigma, n_xyz, r)

	Create Gaussian convoluted image from a set of bead positions

	Parameter
	---------

	pos_x:  array_like (float), shape=(n_bead)
		Bead position along x dimension

	pos_y:  array_like (float), shape=(n_bead)
		Bead position along y dimension

	std:  float
		Standard deviation of Gaussian distribution

	n_xyz:  tuple (int); shape(n_dim)
		Number of pixels in each image dimension

	r:  array_like (float); shape=(n_x, n_y)
		Matrix of radial distances between pixels

	Returns
	-------

	histogram:  array_like (int); shape=(n_x, n_y)
		Discretised distribution of pos_x and pos_y

	image:  array_like (float); shape=(n_x, n_y)
		Convoluted SHG imitation image

	"""

	n_dim = len(n_xyz)

	"Discretise data"
	histogram, edges = np.histogramdd(pos.T, bins=n_xyz)

	if n_dim == 2: 
		histogram = histogram.T
	elif n_dim == 3: 
		histogram = reorder_array(histogram)
		r = reorder_array(r)

	"Get indicies and intensity of non-zero histogram grid points"
	indices = np.argwhere(histogram)
	intensity = histogram[np.where(histogram)]

	"Generate blank image"
	image = np.zeros(n_xyz[:2])

	for i, index in enumerate(indices):

		"""
		"Performs filtered mapping"
		if n_dim == 2:
			r_cut_shift = move_array_centre(r_cut, index[::-1])
			filter_shift = move_array_centre(filter_, index[::-1])
		elif n_dim == 3:
			r_cut_shift = move_array_centre(r_cut[index[0]], index[1:])
			filter_shift = move_array_centre(filter_[index[0]], index[1:])

		pixels = np.where(r_cut_shift)
		image[pixels] += gaussian(r_cut_shift[pixels].flatten(), 0, std) * intensity[i]
		"""		
		"Performs the full mapping"
		if n_dim == 2: r_shift = move_array_centre(r, index[::-1])
		elif n_dim == 3: r_shift = move_array_centre(r[index[0]], index[1:])
		image += np.reshape(gaussian(r_shift.flatten(), 0, std), n_xyz[:2]) * intensity[i]

	image = image.T

	return histogram, image


def fibre_align(histogram, std, n_xyz, dxdydz, r_cut, non_zero):
	"""
	create_image(pos_x, pos_y, sigma, n_x, n_y, r_cut, non_zero)

	Create Gaussian convoluted image from a set of bead positions

	Parameter
	---------

	histogram:  array_like (int); shape=(n_x, n_y)
		Discretised distribution of pos_x and pos_y

	std:  float
		Standard deviation of Gaussian distribution

	n_xyz:  tuple (int); shape(n_dim)
		Number of pixels in each image dimension

	dxdydz:  array_like (float); shape=(n_x, n_y, n_z)
		Matrix of distances along x y and z axis in pixels with cutoff radius applied

	r_cut:  array_like (float); shape=(n_x, n_y)
		Matrix of radial distances between pixels with cutoff radius applied

	non_zero:  array_like (float); shape=(n_x, n_y)
		Filter representing indicies to use in convolution

	Returns
	-------

	dx_grid:  array_like (float); shape=(n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	"""

	"Get indicies and intensity of non-zero histogram grid points"
	indices = np.argwhere(histogram)
	intensity = histogram[np.where(histogram)]
	n_dim = len(n_xyz)

	if n_dim == 3: 
		r_cut = reorder_array(r_cut)
		non_zero = reorder_array(non_zero)
		dxdydz = np.moveaxis(dxdydz, (0, 3, 1, 2), (0, 1, 2, 3))

	n_dim = len(n_xyz)
	"Generate blank image"
	dx_grid = np.zeros(n_xyz[:2])
	dy_grid = np.zeros(n_xyz[:2])

	for i, index in enumerate(indices):
	
		if n_dim == 2:
			r_cut_shift = move_array_centre(r_cut, index)
			non_zero_shift = move_array_centre(non_zero, index)
			dx_shift = move_array_centre(dxdydz[0], index)
			dy_shift = move_array_centre(dxdydz[1], index)

		elif n_dim == 3:

			r_cut_shift = move_array_centre(r_cut[-index[0]], index[1:])
			non_zero_shift = move_array_centre(non_zero[-index[0]], index[1:])
			dx_shift = move_array_centre(dxdydz[0][-index[0]], index[1:])
			dy_shift = move_array_centre(dxdydz[1][-index[0]], index[1:])
			
		dx_grid[np.where(non_zero_shift)] += (dx_gaussian(r_cut_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dx_shift[np.where(non_zero_shift)].flatten() / r_cut_shift[np.where(non_zero_shift)].flatten())
		dy_grid[np.where(non_zero_shift)] += (dx_gaussian(r_cut_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dy_shift[np.where(non_zero_shift)].flatten() / r_cut_shift[np.where(non_zero_shift)].flatten())

	dx_grid = dx_grid.T
	dy_grid = dy_grid.T

	return dx_grid, dy_grid


def make_png(file_name, fig_dir, image, res, sharp, cell_dim, itype='MD'):
	"""
	make_gif(file_name, fig_dir, image, res, sharp, cell_dim, itype='MD')

	Create a png out of image data

	Parameters
	----------

	file_name:  str
		Name of gif file to be created

	fig_dir:  str
		Directory of figure pngs to use in gif

	image:  array_like (float); shape=(n_y, n_x)
		SHG image to be converted into png

	res:  float
		Parameter determining resolution of SHG images

	sharp:  float
		Parameter determining sharpness of SHG images

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	itype:  str (optional)
		Type of figure to make, scatter plot ('MD') or imshow ('SHG')

	"""

	n_dim = cell_dim.shape[0]

	if itype.upper() == 'MD':
		if n_dim == 2:
			fig, ax = plt.subplots(figsize=(cell_dim[0]/4, cell_dim[1]/4))
			plt.scatter(image[0], image[1])
			plt.xlim(0, cell_dim[0])
			plt.ylim(0, cell_dim[1])
		elif n_dim == 3:
			fig = plt.figure(figsize=(cell_dim[0]/4, cell_dim[1]/4))
			ax = plt3d.Axes3D(fig)
			ax.scatter(image[0], image[1], image[2], s=2*vdw_param[0]*res / l_conv, zdir='y')
			ax.set_xlim3d([0.0, cell_dim[0]])
			ax.set_ylim3d([0.0, cell_dim[2]])
			ax.set_zlim3d([0.0, cell_dim[1]])
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
	"""
	make_gif(file_name, fig_dir, gif_dir, n_frame, images, res, sharp, cell_dim, itype='MD')

	Create a gif out of a series n_frame png figures

	Parameters
	----------

	file_name:  str
		Name of gif file to be created

	fig_dir:  str
		Directory of figure pngs to use in gif

	gif_dir:  str
		Directory of gif to be created

	n_frame:  int
		Number of frames to include in gif

	images:  array_like (float); shape=(n_frame, n_y, n_x)
		SHG images to be converted into pngs

	res:  float
		Parameter determining resolution of SHG images

	sharp:  float
		Parameter determining sharpness of SHG images

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	itype:  str (optional)
		Type of figure to make, scatter plot ('MD') or imshow ('SHG')

	"""

	import imageio

	image_list = []
	file_name_plot = '{}_{}_{}'.format(file_name, res, sharp)

	for frame in range(n_frame):
		#if not os.path.exists('{}/{}_{}_ISM.png'.format(fig_dir, file_name_plot, frame)):
		make_png("{}_{}".format(file_name_plot, frame), fig_dir, images[frame], res, sharp, cell_dim, itype)
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

	pad = int(area/2 - 1)

	for n in range(n_sample):
		start_x = np.random.randint(pad, n_x - pad)
		start_y = np.random.randint(pad, n_y - pad) 
		
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


def analysis(current_dir, dir_path):

	n_dim, n_step, file_names, sim_param, fibril_para, analysis_param = setup.read_shell_input(current_dir, dir_path)

	param_file_name, pos_file_name, traj_file_name, restart_file_name, output_file_name = file_names
	gif_file_name, res, sharp, skip = analysis_param

	print("Loading parameter file {}.pkl".format(param_file_name))
	param_file = ut.read_param_file(param_file_name)
	vdw_param = param_file['vdw_param']
	rc = param_file['rc']
	l_conv = param_file['l_conv']
	bond_matrix = param_file['bond_matrix']
	kBT = param_file['kBT']

	print("Loading output file {}".format(output_file_name))
	tot_energy, tot_temp = ut.load_npy(output_file_name)

	print("Loading trajectory file {}.npy".format(traj_file_name))
	tot_pos = ut.load_npy(traj_file_name)
	n_frame = tot_pos.shape[0]
	n_bead = tot_pos.shape[1]
	cell_dim = tot_pos[0][-1]

	n_xyz = tuple(np.array(cell_dim * res, dtype=int))

	gif_dir = current_dir + '/gif'
	if not os.path.exists(gif_dir): os.mkdir(gif_dir)
	fig_dir = current_dir + '/fig'
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)

	fig_name = traj_file_name.split('/')[-1]

	print('Creating Energy figure {}/{}_energy.png'.format(fig_dir, fig_name))
	plt.figure(0)
	plt.title('Energy')
	plt.plot(tot_energy / n_bead)
	plt.xlabel(r'step')
	plt.ylabel(r'Energy / bead')
	plt.savefig('{}/{}_energy.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print('Creating Temperature figure {}/{}_temp.png'.format(fig_dir, fig_name))
	plt.figure(1)
	plt.title('Temperature / kBT')
	plt.plot(tot_temp / kBT)
	plt.xlabel(r'step')
	plt.ylabel(r'Temp / kBT')
	plt.savefig('{}/{}_temp.png'.format(fig_dir, fig_name), bbox_inches='tight')

	

	n_image = int(n_frame/skip)
	sample_l = 150 / l_conv
	n_sample = 20
	area = int(np.min([sample_l, np.min(cell_dim[:2])]) / l_conv * res)

	image_md = np.moveaxis([tot_pos[n][:-1] for n in range(0, n_frame, skip)], 2, 1)

	"Generate Gaussian convoluted images and intensity derivatives"
	image_shg, dx_shg, dy_shg = shg_images(image_md, 2 * vdw_param[0] / (l_conv * sharp) * res, n_xyz, 2 * rc / (l_conv * sharp) * res)
	"Calculate intensity orientational vector n for each pixel"
	n_vector = form_n_vector(dx_shg, dy_shg)
	"Sample average orientational anisotopy"
	eigval_shg, eigvec_shg = alignment_analysis(n_vector, area, n_sample)

	q = reorder_array(eigval_shg)
	q = q[1] - q[0]

	print('Mean anistoropy = {}'.format(np.mean(q)))

	make_gif(gif_file_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, res, sharp, cell_dim, 'SHG')
	#make_gif(gif_file_name + '_MD', fig_dir, gif_dir, n_image, image_md, res, sharp, cell_dim, 'MD')

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

