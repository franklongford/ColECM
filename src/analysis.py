"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os

import utilities as ut
import setup

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


def create_image(pos, std, n_xyz, r):
	"""
	create_image(pos_x, pos_y, sigma, n_xyz, r)

	Create Gaussian convoluted image from a set of bead positions

	Parameter
	---------

	pos:  array_like (float), shape=(n_bead)
		Bead positions along n_dim dimension

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


def fibril_align(histogram, std, n_xyz, dxdydz, r, non_zero):
	"""
	create_image(pos_x, pos_y, sigma, n_x, n_y, r, non_zero)

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
		r = reorder_array(r)
		#r_cut = reorder_array(r_cut)
		non_zero = reorder_array(non_zero)
		dxdydz = np.moveaxis(dxdydz, (0, 3, 1, 2), (0, 1, 2, 3))

	n_dim = len(n_xyz)
	"Generate blank image"
	dx_grid = np.zeros(n_xyz[:2])
	dy_grid = np.zeros(n_xyz[:2])

	for i, index in enumerate(indices):
	
		"""
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

		"""
		if n_dim == 2:
			r_shift = move_array_centre(r, index)
			non_zero_shift = move_array_centre(non_zero, index)
			dx_shift = move_array_centre(dxdydz[0], index)
			dy_shift = move_array_centre(dxdydz[1], index)

		elif n_dim == 3:

			r_shift = move_array_centre(r[-index[0]], index[1:])
			non_zero_shift = move_array_centre(non_zero[-index[0]], index[1:])
			dx_shift = move_array_centre(dxdydz[0][-index[0]], index[1:])
			dy_shift = move_array_centre(dxdydz[1][-index[0]], index[1:])
			
		dx_grid[np.where(non_zero_shift)] += (dx_gaussian(r_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dx_shift[np.where(non_zero_shift)].flatten() / r_shift[np.where(non_zero_shift)].flatten())
		dy_grid[np.where(non_zero_shift)] += (dx_gaussian(r_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dy_shift[np.where(non_zero_shift)].flatten() / r_shift[np.where(non_zero_shift)].flatten())

		#"""


	dx_grid = dx_grid.T
	dy_grid = dy_grid.T

	return dx_grid, dy_grid


def shg_images(traj, sigma, n_xyz, cut):
	"""
	shg_images(traj, sigma, n_xyz_md, cut)

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

	image_shg = np.zeros((n_image,) +  n_xyz[:2][::-1])
	dx_shg = np.zeros((n_image,) +  n_xyz[:2][::-1])
	dy_shg = np.zeros((n_image,) +  n_xyz[:2][::-1])

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
	#r_cut = np.zeros(n_xyz)
	#r_cut[cutoff] += np.sqrt(r2[cutoff])

	for image in range(n_image):
		sys.stdout.write(" Processing image {} out of {}\r".format(image, n_image))
		sys.stdout.flush()
		
		hist, image_shg[image] = create_image(traj[image], sigma, n_xyz, r)
		dx_shg[image], dy_shg[image] = fibril_align(hist, sigma, n_xyz, dxdydz, r, non_zero)

	return image_shg, dx_shg, dy_shg



def make_png(file_name, fig_dir, image, bonds, res, sharp, cell_dim, itype='MD'):
	"""
	make_gif(file_name, fig_dir, image, bond, res, sharp, cell_dim, itype='MD')

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
			for bond in bonds:plt.plot(image[0][bond], image[1][bond], linestyle='dashed')
			plt.xlim(0, cell_dim[0])
			plt.ylim(0, cell_dim[1])
		elif n_dim == 3:
			plt.close('all')
			fig = plt.figure(figsize=(cell_dim[0]/4, cell_dim[1]/4))
			ax = plt3d.Axes3D(fig)
			ax.scatter(image[0], image[1], image[2], zdir='y')
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


def make_gif(file_name, fig_dir, gif_dir, n_frame, images, param, cell_dim, itype='MD'):
	"""
	make_gif(file_name, fig_dir, gif_dir, n_frame, images, bond_matrix, res, sharp, cell_dim, itype='MD')

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
	file_name_plot = '{}_{}_{}'.format(file_name, param['res'], param['sharp'])
	indices = np.arange(param['n_bead']).reshape((param['n_fibril'], param['l_fibril']))
	
	for frame in range(n_frame):
		#if not os.path.exists('{}/{}_{}_ISM.png'.format(fig_dir, file_name_plot, frame)):
		make_png("{}_{}".format(file_name_plot, frame), gif_dir, images[frame], indices, param['res'], param['sharp'], cell_dim, itype)
		image_list.append('{}/{}_{}_ISM.png'.format(gif_dir, file_name_plot, frame))

	file_name_gif = '{}_{}_{}_{}'.format(file_name, param['res'], param['sharp'], n_frame)
	file_path_name = '{}/{}.gif'.format(gif_dir, file_name_gif)

	with imageio.get_writer(file_path_name, mode='I', duration=0.3, format='GIF') as writer:
		for filename in image_list:
			image = imageio.imread(filename)
			writer.append_data(image)
			os.remove(filename)


def fibre_vector_analysis(traj, cell_dim, param):
	"""
	fibre_vector_analysis(traj, cell_dim, param)

	Parameters
	----------

	traj:  array_like (float); shape=(n_images, n_dim, n_bead)
			Array of sampled configurations from a simulation

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	param:

	Returns
	-------

	tot_vectors, tot_mag

	"""
	
	n_image = traj.shape[0]
	n_bond = int(np.sum(np.triu(param['bond_matrix'])))
	bond_index_half = np.argwhere(np.triu(param['bond_matrix']))
	indices_half = ut.create_index(bond_index_half)
	bond_list = np.zeros((param['n_dim'], n_bond))

	tot_mag = np.zeros((n_image, param['n_fibril']))
	#tot_theta = np.zeros((n_image, param['n_fibril']))

	#tot_mag = np.zeros((n_image, n_bond))
	tot_theta = np.zeros((n_image, n_bond))

	for image, pos in enumerate(traj):

		distances = ut.get_distances(pos.T, cell_dim)
		for i in range(param['n_dim']): bond_list[i] = distances[i][indices_half]

		bead_vectors = bond_list.T
		mag_vectors = np.sqrt(np.sum(bead_vectors**2, axis=1))
		#cos_theta = bead_vectors.T[0] / mag_vectors
		sin_theta = bead_vectors.T[1] / mag_vectors

		norm_vectors = bead_vectors / np.resize(mag_vectors, bead_vectors.shape)
		fib_vectors = np.sum(norm_vectors.reshape(param['l_fibril']-1, param['n_fibril'], param['n_dim']), axis=0)
		mag_vectors = np.sqrt(np.sum(fib_vectors**2, axis=1))
		#cos_theta = fib_vectors.T[0] / mag_vectors
		#sin_theta = fib_vectors.T[1] / mag_vectors

		tot_theta[image] += np.arcsin(sin_theta) * 360 / np.pi
		#tot_theta[image] += np.arccos(cos_theta) * 360 / np.pi
		tot_mag[image] += mag_vectors / (param['l_fibril']-1)

	return tot_theta, tot_mag


def form_nematic_tensor(dx_shg, dy_shg):
	"""
	form_nematic_tensor(dx_shg, dy_shg)

	Create local nematic tensor n for each pixel in dx_shg, dy_shg

	Parameters
	----------

	dx_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	Returns
	-------

	n_vector:  array_like (float); shape(nframe, 4, n_y, n_x)
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


def nematic_tensor_analysis(n_vector, area, n_sample):
	"""
	nematic_tensor_analysis(n_vector, area, n_frame, n_sample)

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

	pad = int(area / 2 - 1)

	for n in range(n_sample):

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad

		cut_n_vector = n_vector[:, :, start_y-pad: start_y+pad, 
					      start_x-pad: start_x+pad]

		av_n = np.reshape(np.mean(cut_n_vector, axis=(2, 3)), (n_frame, 2, 2))

		for frame in range(n_frame):
	
			eig_val, eig_vec = np.linalg.eigh(av_n[frame])

			av_eigval[frame][n] = eig_val
			av_eigvec[frame][n] = eig_vec

	return av_eigval, av_eigvec
	

def fourier_transform_analysis(image_shg, area, n_sample):
	"""
	fourier_transform_analysis(image_shg, area, n_sample)

	Calculates fourier amplitude spectrum of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	angles:  array_like (float); shape=(n_bins)
		Angles corresponding to fourier amplitudes

	fourier_spec:  array_like (float); shape=(n_bins)
		Average Fouier amplitudes of FT of image_shg

	"""

	n_frame = image_shg.shape[0]
	n_y = image_shg.shape[1]
	n_x = image_shg.shape[2]

	pad = area // 2

	cut_image = image_shg[0, : area, : area]

	image_fft = np.fft.fft2(cut_image)
	image_fft[0][0] = 0
	image_fft = np.fft.fftshift(image_fft)
	average_fft = np.zeros(image_fft.shape, dtype=complex)

	fft_angle = np.angle(image_fft, deg=True)
	angles = np.unique(fft_angle)
	fourier_spec = np.zeros(angles.shape)
	
	n_bins = fourier_spec.size

	for n in range(n_sample):

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad

		cut_image = image_shg[:, start_y-pad: start_y+pad, 
					 start_x-pad: start_x+pad]

		for frame in range(n_frame):

			image_fft = np.fft.fft2(cut_image[frame])
			image_fft[0][0] = 0
			average_fft += np.fft.fftshift(image_fft) / (n_frame * n_sample)	

	for i in range(n_bins):
		indices = np.where(fft_angle == angles[i])
		fourier_spec[i] += np.sum(np.abs(average_fft[indices])) / 360

	return angles, fourier_spec


def plot_gallery(n, title, images, n_col, n_row, image_shape):

	plt.figure(n, figsize=(2. * n_col, 2.26 * n_row))
	plt.suptitle(title, size=16)

	for i, comp in enumerate(images):
		plt.subplot(n_row, n_col, i + 1)
		vmax = max(comp.max(), -comp.min())
		plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
			   interpolation='nearest',
			   vmin=-vmax, vmax=vmax)
		plt.xticks(())
		plt.yticks(())

	plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


def nmf_analysis(image_shg, area, n_sample, n_components):
	"""
	nmf_analysis(image_shg, area, n_sample)

	Calculates non-negative matrix factorisation of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample


	Returns
	-------

	"""

	from sklearn.decomposition import NMF
	from sklearn.datasets import fetch_olivetti_faces

	n_frame = image_shg.shape[0]
	n_y = image_shg.shape[1]
	n_x = image_shg.shape[2]
	rng = np.random.RandomState(0)

	model = NMF(n_components=n_components, init='random', random_state=0)
	pad = area // 2

	for n in range(n_sample):

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad-1
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad-1

		cut_image = image_shg[:, start_y-pad: start_y+pad, 
					 start_x-pad: start_x+pad].reshape(n_frame, area**2)

		model.fit(cut_image)

		nmf_components = model.components_

	return nmf_components


def animate(n):
	plt.title('Frame {}'.format(n))
	sc.set_offsets(np.c_[tot_pos[n][0], tot_pos[n][1]])


def heatmap_animation(n):
	plt.title('Frame {}'.format(n))
	ax.pcolor(image_pos[n], cmap='viridis')


def analysis(current_dir, input_file_name=False):

	print("\n " + " " * 15 + "----Beginning Analysis----\n")

	sim_dir = current_dir + '/sim/'
	gif_dir = current_dir + '/gif'
	if not os.path.exists(gif_dir): os.mkdir(gif_dir)
	fig_dir = current_dir + '/fig'
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)

	ow_shg = ('-ow_shg' in sys.argv)
	mk_gif = ('-mk_gif' in sys.argv)

	file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name)
	fig_name = file_names['gif_file_name'].split('/')[-1]

	keys = ['l_conv', 'res', 'sharp', 'skip']
	print("\n Analysis Parameters found:")
	for key in keys: print(" {:<15s} : {}".format(key, param[key]))	

	print("\n Loading output file {}{}".format(sim_dir, file_names['output_file_name']))
	tot_energy, tot_temp, tot_press = ut.load_npy(sim_dir + file_names['output_file_name'])

	print('\n Creating Energy time series figure {}/{}_energy_time.png'.format(fig_dir, fig_name))
	plt.figure(0)
	plt.title('Energy Time Series')
	plt.plot(tot_energy * param['l_fibril'] / param['n_bead'], label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Energy per fibril')
	plt.legend()
	plt.savefig('{}/{}_energy_time.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Energy histogram figure {}/{}_energy_hist.png'.format(fig_dir, fig_name))
	plt.figure(1)
	plt.title('Energy Histogram')
	plt.hist(tot_energy * param['l_fibril'] / param['n_bead'], bins='auto', density=True, label=fig_name)
	plt.xlabel(r'Energy per fibril')
	plt.legend()
	plt.savefig('{}/{}_energy_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Temperature time series figure {}/{}_temp_time.png'.format(fig_dir, fig_name))
	plt.figure(2)
	plt.title('Temperature Time Series')
	plt.plot(tot_temp, label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Temp (kBT)')
	plt.legend()
	plt.savefig('{}/{}_temp_time.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Temperature histogram figure {}/{}_temp_hist.png'.format(fig_dir, fig_name))
	plt.figure(3)
	plt.title('Temperature Histogram')
	plt.hist(tot_temp, bins='auto', density=True, label=fig_name)
	plt.xlabel(r'Temp (kBT)')
	plt.legend()
	plt.savefig('{}/{}_temp_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Pressure time series figure {}/{}_press_time.png'.format(fig_dir, fig_name))
	plt.figure(4)
	plt.title('Pressure Time Series')
	plt.plot(tot_press, label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Pressure')
	plt.legend()
	plt.savefig('{}/{}_press_time.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Pressure histogram figure {}/{}_press_hist.png'.format(fig_dir, fig_name))
	plt.figure(5)
	plt.title('Pressure Histogram')
	plt.hist(tot_press, bins='auto', density=True, label=fig_name)
	plt.xlabel(r'Pressure')
	plt.legend()
	plt.savefig('{}/{}_press_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print("\n Loading trajectory file {}{}.npy".format(sim_dir, file_names['traj_file_name']))
	tot_pos = ut.load_npy(sim_dir + file_names['traj_file_name'])

	n_frame = tot_pos.shape[0]
	n_image = int(n_frame / param['skip'])
	cell_dim = tot_pos[0][-1]
	n_xyz = tuple(np.array(cell_dim * param['l_conv'] * param['res'], dtype=int))
	
	conv = param['l_conv'] / param['sharp'] * param['res']

	image_md = np.moveaxis([tot_pos[n][:-1] for n in range(0, n_frame)], 2, 1)

	"Perform Fibre Vector analysis"
	tot_theta, tot_mag = fibre_vector_analysis(image_md, cell_dim, param)
	hist, bin_edges = np.histogram(tot_theta.flatten(), bins='auto', density=True)

	print('\n Modal Vector angle  = {:>6.4f}'.format(bin_edges[np.argmax(hist)]))
	print(' Mean Fibril RMS = {:>6.4f}'.format(np.mean(tot_mag)))
	print(' Expected Random Walk RMS = {:>6.4f}'.format(1. / np.sqrt(param['l_fibril']-1)))

	print(' Creating Vector Magnitude histogram figure {}/{}_vec_mag_hist.png'.format(fig_dir, fig_name))
	plt.figure(7)
	plt.title('Vector Magnitude Histogram')
	plt.hist(tot_mag.flatten(), bins='auto', density=True, label=fig_name)
	plt.xlabel(r'$|R|$')
	#plt.axis([0, 2.0, 0, 3.0])
	plt.legend()
	plt.savefig('{}/{}_vec_mag_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Vector Angular histogram figure {}/{}_vec_ang_hist.png'.format(fig_dir, fig_name))
	plt.figure(8)
	plt.title('Vector Angle Histogram')
	plt.hist(tot_theta.flatten(), bins='auto', density=True, label=fig_name)
	plt.xlabel(r'$\theta$')
	plt.xlim(-180, 180)
	plt.legend()
	plt.savefig('{}/{}_vec_ang_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')


	image_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_{}_{}_{}_image_shg'.format(n_frame, param['res'], param['sharp'])
	dx_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_{}_{}_{}_dx_shg'.format(n_frame, param['res'], param['sharp'])
	dy_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_{}_{}_{}_dy_shg'.format(n_frame, param['res'], param['sharp'])

	if not ow_shg:
		try:
			image_shg = ut.load_npy(sim_dir + image_file_name, range(0, n_frame, param['skip']))	
			dx_shg = ut.load_npy(sim_dir + dx_file_name, range(0, n_frame, param['skip']))
			dy_shg = ut.load_npy(sim_dir + dy_file_name, range(0, n_frame, param['skip']))
		except: ow_shg = True
	
	if ow_shg:
		"Generate Gaussian convoluted images and intensity derivatives"
		image_shg, dx_shg, dy_shg = shg_images(image_md, param['vdw_sigma'] * conv, n_xyz, 2 * param['rc'] * conv)

		print("\n Saving image files {}".format(file_names['output_file_name']))
		ut.save_npy(sim_dir + image_file_name, image_shg)
		ut.save_npy(sim_dir + dx_file_name, dx_shg)
		ut.save_npy(sim_dir + dy_file_name, dy_shg)

		image_shg = np.array([image_shg[i] for i in range(0, n_frame, param['skip'])])
		dx_shg = np.array([dx_shg[i] for i in range(0, n_frame, param['skip'])])
		dy_shg = np.array([dy_shg[i] for i in range(0, n_frame, param['skip'])])


	fig_name += '_{}_{}'.format(param['res'], param['sharp'])

	"Perform Nematic Tensor Analysis"

	l_sample = 30
	n_sample = 10
	area_sample = int(2 * (np.min((l_sample,) + image_shg.shape[1:]) // 2))

	n_tensor = form_nematic_tensor(dx_shg, dy_shg)
	"Sample average orientational anisotopy"
	eigval_shg, eigvec_shg = nematic_tensor_analysis(n_tensor, area_sample, n_sample)

	q = reorder_array(eigval_shg)
	q = q[1] - q[0]

	print('\n Mean anistoropy = {:>6.4f}'.format(np.mean(q)))
	anis_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_anis'
	print(" Saving anisotropy file {}".format(file_names['output_file_name']))
	ut.save_npy(sim_dir + anis_file_name, q)

	print(' Creating Anisotropy time series figure {}/{}_anis_time.png'.format(fig_dir, fig_name))
	plt.figure(9)
	plt.title('Anisotropy Time Series')
	plt.plot(np.mean(q, axis=1), label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Anisotropy')
	plt.ylim(0, 1)
	plt.legend()
	plt.savefig('{}/{}_anis_time.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Anisotropy histogram figure {}/{}_anis_hist.png'.format(fig_dir, fig_name))
	plt.figure(10)
	plt.title('Anisotropy Histogram')
	plt.hist(np.mean(q, axis=1), bins='auto', density=True, label=fig_name)
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig('{}/{}_anis_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')


	"Perform Fourier Analysis"
	angles, fourier_spec = fourier_transform_analysis(image_shg, area_sample, n_sample)

	#angles = angles[len(angles)//2:]
	#fourier_spec = 2 * fourier_spec[len(fourier_spec)//2:]

	print('\n Modal Fourier Amplitude  = {:>6.4f}'.format(angles[np.argmax(fourier_spec)]))
	print(' Fourier Amplitudes Range   = {:>6.4f}'.format(np.max(fourier_spec)-np.min(fourier_spec)))
	print(' Fourier Amplitudes Std Dev = {:>6.4f}'.format(np.std(fourier_spec)))

	print(' Creating Fouier Angle Spectrum figure {}/{}_fourier.png'.format(fig_dir, fig_name))
	plt.figure(11)
	plt.title('Fourier Angle Spectrum')
	plt.plot(angles, fourier_spec, label=fig_name)
	plt.xlabel(r'Angle (deg)')
	plt.ylabel(r'Amplitude')
	plt.xlim(-180, 180)
	plt.ylim(0, 1.00)
	plt.legend()
	plt.savefig('{}/{}_fourier.png'.format(fig_dir, fig_name), bbox_inches='tight')


	"Perform Non-Negative Matrix Factorisation"
	n_components = 9
	pad = int(area_sample / 2 - 1)

	nmf_components = nmf_analysis(image_shg, area_sample, n_sample, n_components)

	print('\n Creating NMF Gallery {}/{}_nmf.png'.format(fig_dir, fig_name))
	plot_gallery(12, 'NMF Main Components', nmf_components[:n_components], np.sqrt(n_components), np.sqrt(n_components), (area_sample, area_sample))
	plt.savefig('{}/{}_nmf.png'.format(fig_dir, fig_name), bbox_inches='tight')


	"Make Gif of SHG Images"
	if ow_shg or mk_gif:
		print('\n Making Simulation SHG Gif {}/{}.gif'.format(fig_dir, fig_name))
		make_gif(fig_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, param, cell_dim * param['l_conv'], 'SHG')

	#print(' Making Simulation MD Gif {}/{}.gif'.format(fig_dir, fig_name))
	#make_gif(fig_name + '_MD', fig_dir, gif_dir, n_image, image_md * param['l_conv'], param, cell_dim * param['l_conv'], 'MD')

