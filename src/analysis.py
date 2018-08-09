"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import numpy as np
import scipy as sp
from scipy import signal

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os

import utilities as ut
import setup


def print_thermo_results(fig_dir, fig_name, tot_energy, tot_temp, tot_press):


	print('\n Creating Energy time series figure {}{}_energy_time.png'.format(fig_dir, fig_name))
	plt.figure(0)
	plt.title('Energy Time Series')
	plt.plot(tot_energy, label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Energy per fibril')
	plt.legend()
	plt.savefig(fig_dir + fig_name + '_energy_time.png', bbox_inches='tight')

	print(' Creating Energy histogram figure {}{}_energy_hist.png'.format(fig_dir, fig_name))
	plt.figure(1)
	plt.title('Energy Histogram')
	plt.hist(tot_energy, bins='auto', density=True, label=fig_name)#, range=[np.mean(tot_energy) - np.std(tot_energy), np.mean(tot_energy) + np.std(tot_energy)])
	plt.xlabel(r'Energy per fibril')
	plt.legend()
	plt.savefig(fig_dir + fig_name + '_energy_hist.png', bbox_inches='tight')

	print(' Creating Temperature time series figure {}{}_temp_time.png'.format(fig_dir, fig_name))
	plt.figure(2)
	plt.title('Temperature Time Series')
	plt.plot(tot_temp, label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Temp (kBT)')
	plt.legend()
	plt.savefig(fig_dir + fig_name + '_temp_time.png', bbox_inches='tight')

	print(' Creating Temperature histogram figure {}{}_temp_hist.png'.format(fig_dir, fig_name))
	plt.figure(3)
	plt.title('Temperature Histogram')
	plt.hist(tot_temp, bins='auto', density=True, label=fig_name)#, range=[np.mean(tot_temp) - np.std(tot_temp), np.mean(tot_temp) + np.std(tot_temp)])
	plt.xlabel(r'Temp (kBT)')
	plt.legend()
	plt.savefig(fig_dir + fig_name + '_temp_hist.png', bbox_inches='tight')

	print(' Creating Pressure time series figure {}{}_press_time.png'.format(fig_dir, fig_name))
	plt.figure(4)
	plt.title('Pressure Time Series')
	plt.plot(tot_press, label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Pressure')
	plt.legend()
	plt.savefig(fig_dir + fig_name + '_press_time.png', bbox_inches='tight')

	print(' Creating Pressure histogram figure {}{}_press_hist.png'.format(fig_dir, fig_name))
	plt.figure(5)
	plt.title('Pressure Histogram')
	plt.hist(tot_press, bins='auto', density=True, range=[np.mean(tot_press) - np.std(tot_press), np.mean(tot_press) + np.std(tot_press)], label=fig_name)
	plt.xlabel(r'Pressure')
	plt.legend()
	plt.savefig(fig_dir + fig_name + '_press_hist.png', bbox_inches='tight')
	plt.close('all')


def print_vector_results(fig_dir, fig_name, param, tot_mag, tot_theta):

	hist, bin_edges = np.histogram(tot_theta.flatten(), bins='auto', density=True)

	print('\n Modal Vector angle  = {:>6.4f}'.format(bin_edges[np.argmax(hist)]))
	print(' Mean Fibril RMS = {:>6.4f}'.format(np.mean(tot_mag)))
	print(' Expected Random Walk RMS = {:>6.4f}'.format(1. / np.sqrt(param['l_fibril']-1)))

	print(' Creating Vector Magnitude histogram figure {}{}_vec_mag_hist.png'.format(fig_dir, fig_name))
	plt.figure(7)
	plt.title('Vector Magnitude Histogram')
	plt.hist(tot_mag.flatten(), bins='auto', density=True, label=fig_name)
	plt.xlabel(r'$|R|$')
	#plt.axis([0, 2.0, 0, 3.0])
	plt.legend()
	plt.savefig('{}{}_vec_mag_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Vector Angular histogram figure {}{}_vec_ang_hist.png'.format(fig_dir, fig_name))
	plt.figure(8)
	plt.title('Vector Angle Histogram')
	plt.hist(tot_theta.flatten(), bins='auto', density=True, label=fig_name)
	plt.xlabel(r'$\theta$')
	plt.xlim(-180, 180)
	plt.legend()
	plt.savefig('{}{}_vec_ang_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def print_anis_results(fig_dir, fig_name, q):

	print('\n Mean anistoropy = {:>6.4f}'.format(np.mean(q)))

	print(' Creating Anisotropy time series figure {}{}_anis_time.png'.format(fig_dir, fig_name))
	plt.figure(9)
	plt.title('Anisotropy Time Series')
	plt.plot(q, label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'Anisotropy')
	plt.ylim(0, 1)
	plt.legend()
	plt.savefig('{}{}_anis_time.png'.format(fig_dir, fig_name), bbox_inches='tight')

	print(' Creating Anisotropy histogram figure {}{}_anis_hist.png'.format(fig_dir, fig_name))
	plt.figure(10)
	plt.title('Anisotropy Histogram')
	plt.hist(q, bins='auto', density=True, label=fig_name)
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig('{}{}_anis_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def print_fourier_results(fig_dir, fig_name, angles, fourier_spec, sdi):

	print('\n Modal Fourier Amplitude  = {:>6.4f}'.format(angles[np.argmax(fourier_spec)]))
	print(' Fourier Amplitudes Range   = {:>6.4f}'.format(np.max(fourier_spec)-np.min(fourier_spec)))
	print(' Fourier Amplitudes Std Dev = {:>6.4f}'.format(np.std(fourier_spec)))
	print(' Fourier Mean SDI = {:>6.4f}'.format(np.mean(sdi)))

	print(' Creating Fouier Angle Spectrum figure {}{}_fourier.png'.format(fig_dir, fig_name))
	plt.figure(11)
	plt.title('Fourier Angle Spectrum')
	plt.plot(angles, fourier_spec, label=fig_name)
	plt.xlabel(r'Angle (deg)')
	plt.ylabel(r'Amplitude')
	plt.xlim(-180, 180)
	#plt.ylim(0, 0.25)
	plt.legend()
	plt.savefig('{}{}_fourier.png'.format(fig_dir, fig_name), bbox_inches='tight')

	plt.figure(12)
	plt.title('Fourier SDI')
	plt.plot(sdi, label=fig_name)
	plt.xlabel(r'step')
	plt.ylabel(r'SDI')
	plt.legend()
	plt.savefig('{}{}_sdi.png'.format(fig_dir, fig_name), bbox_inches='tight')

	plt.close('all')


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
		histogram = ut.reorder_array(histogram)
		r = ut.reorder_array(r)

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
		if n_dim == 2: r_shift = ut.move_array_centre(r, index[::-1])
		elif n_dim == 3: r_shift = ut.move_array_centre(r[index[0]], index[1:])
		image += np.reshape(ut.gaussian(r_shift.flatten(), 0, std), n_xyz[:2]) * intensity[i]

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
		r = ut.reorder_array(r)
		#r_cut = reorder_array(r_cut)
		non_zero = ut.reorder_array(non_zero)
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
			r_shift = ut.move_array_centre(r, index)
			non_zero_shift = ut.move_array_centre(non_zero, index)
			dx_shift = ut.move_array_centre(dxdydz[0], index)
			dy_shift = ut.move_array_centre(dxdydz[1], index)

		elif n_dim == 3:

			r_shift = ut.move_array_centre(r[-index[0]], index[1:])
			non_zero_shift = ut.move_array_centre(non_zero[-index[0]], index[1:])
			dx_shift = ut.move_array_centre(dxdydz[0][-index[0]], index[1:])
			dy_shift = ut.move_array_centre(dxdydz[1][-index[0]], index[1:])
			
		dx_grid[np.where(non_zero_shift)] += (ut.dx_gaussian(r_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dx_shift[np.where(non_zero_shift)].flatten() / r_shift[np.where(non_zero_shift)].flatten())
		dy_grid[np.where(non_zero_shift)] += (ut.dx_gaussian(r_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
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
	plt.savefig('{}{}_ISM.png'.format(fig_dir, file_name), bbox_inches='tight')
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
		image_list.append('{}{}_{}_ISM.png'.format(gif_dir, file_name_plot, frame))

	file_name_gif = '{}_{}_{}_{}'.format(file_name, param['res'], param['sharp'], n_frame)
	file_path_name = '{}{}.gif'.format(gif_dir, file_name_gif)

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
	bond_indices = np.nonzero(np.triu(param['bond_matrix']))
	n_bond = bond_indices[0].shape[0]
	bond_list = np.zeros((param['n_dim'], n_bond))

	tot_mag = np.zeros((n_image, param['n_fibril']))
	#tot_theta = np.zeros((n_image, param['n_fibril']))

	#tot_mag = np.zeros((n_image, n_bond))
	tot_theta = np.zeros((n_image, n_bond))

	for image, pos in enumerate(traj):

		distances = ut.get_distances(pos.T, cell_dim)
		for i in range(param['n_dim']): bond_list[i] = distances[i][bond_indices]

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


def select_samples(full_set, area, n_sample):

	
	n_frame = full_set.shape[0]
	n_y = full_set.shape[1]
	n_x = full_set.shape[2]
	data_set = np.zeros((n_sample, n_frame, area, area))

	pad = area // 2

	indices = np.zeros((n_sample, 2), dtype=int)

	for n in range(n_sample):

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad

		indices[n][0] = start_x
		indices[n][1] = start_y

		data_set[n] = full_set[:, start_y-pad: start_y+pad, 
					  start_x-pad: start_x+pad]

	return data_set.reshape(n_sample * n_frame, area, area), indices


def nematic_tensor_analysis(nem_vector):
	"""
	nematic_tensor_analysis(nem_vector)

	Calculates eigenvalues and eigenvectors of average nematic tensor over area^2 pixels for n_samples

	Parameters
	----------

	nem_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
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

	n_sample = nem_vector.shape[0]
	tot_q = np.zeros(n_sample)

	for n in range(n_sample):
		av_n = np.reshape(np.mean(nem_vector[n], axis=(1, 2)), (2, 2))
		eig_val, eig_vec = np.linalg.eigh(av_n)
		tot_q[n] += (eig_val.T[1] - eig_val.T[0])

	return tot_q


def smart_nematic_tensor_analysis(nem_vector, precision=1E-1):
	"""
	nematic_tensor_analysis(nem_vector)

	Calculates eigenvalues and eigenvectors of average nematic tensor over area^2 pixels for n_samples

	Parameters
	----------

	nem_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
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

	n_sample = nem_vector.shape[0]
	tot_q = np.zeros(n_sample)
	map_shape = nem_vector.shape[2:]

	def rec_search(nem_vector, q):

		image_shape = q.shape
		if image_shape[0] <= 2: return q

		for i in range(2):
			for j in range(2):
				vec_section = nem_vector[:,
									i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
									j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2 ]
				q_section = q[i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
							j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2]

				av_n = np.reshape(np.mean(vec_section, axis=(1, 2)), (2, 2))
				eig_val, eig_vec = np.linalg.eigh(av_n)
				new_q = (eig_val.T[1] - eig_val.T[0])
				old_q = np.mean(q_section)

				if abs(new_q - old_q) >= precision: q_section = rec_search(vec_section, q_section)
				else: q_section = np.ones(vec_section.shape[1:]) * new_q

				q[i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
				  j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2] = q_section

		return q

	for n in range(n_sample):
		vector_map = nem_vector[n]
		q0 = np.zeros(map_shape)
		av_n = np.reshape(np.mean(vector_map, axis=(1, 2)), (2, 2))
		eig_val, eig_vec = np.linalg.eigh(av_n)
		q0 += (eig_val.T[1] - eig_val.T[0])
		q1 = rec_search(vector_map, q0)

		tot_q[n] = np.mean(np.unique(q1))

	return tot_q


def fourier_transform_analysis(image_shg):
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

	n_sample = image_shg.shape[0]

	image_fft = np.fft.fft2(image_shg[0])
	image_fft[0][0] = 0
	image_fft = np.fft.fftshift(image_fft)
	average_fft = np.zeros(image_fft.shape, dtype=complex)
	sdi = np.zeros(n_sample)

	fft_angle = np.angle(image_fft, deg=True)
	angles = np.unique(fft_angle)
	fourier_spec = np.zeros(angles.shape)
	
	n_bins = fourier_spec.size

	for n in range(n_sample):
		image_fft = np.fft.fft2(image_shg[n])
		image_fft[0][0] = 0
		average_fft += np.fft.fftshift(image_fft) / n_sample	
		sdi[n] = np.mean(np.abs(image_fft)) / (np.max(np.abs(image_fft)))

	for i in range(n_bins):
		indices = np.where(fft_angle == angles[i])
		fourier_spec[i] += np.sum(np.abs(average_fft[indices])) / 360

	return angles, fourier_spec, sdi


def animate(n):
	plt.title('Frame {}'.format(n))
	sc.set_offsets(np.c_[tot_pos[n][0], tot_pos[n][1]])


def heatmap_animation(n):
	plt.title('Frame {}'.format(n))
	ax.pcolor(image_pos[n], cmap='viridis')


def analysis(current_dir, input_file_name=False):

	sim_dir = current_dir + '/sim/'
	gif_dir = current_dir + '/gif/'
	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	ow_shg = ('-ow_shg' in sys.argv)
	ow_data = ('-ow_data' in sys.argv)
	mk_gif = ('-mk_gif' in sys.argv)

	print("\n " + " " * 15 + "----Beginning Image Analysis----\n")
	if not os.path.exists(gif_dir): os.mkdir(gif_dir)
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name, verbosity=False)
	fig_name = file_names['gif_file_name'].split('/')[-1]

	keys = ['l_conv', 'res', 'sharp', 'skip', 'l_sample', 'min_sample']
	print("\n Analysis Parameters found:")
	for key in keys: print(" {:<15s} : {}".format(key, param[key]))	

	print("\n Loading output file {}{}".format(sim_dir, file_names['output_file_name']))
	tot_energy, tot_temp, tot_press = ut.load_npy(sim_dir + file_names['output_file_name'])
	
	tot_energy *= param['l_fibril'] / param['n_bead']
	print_thermo_results(fig_dir, fig_name, tot_energy, tot_temp, tot_press)

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
	print_vector_results(fig_dir, fig_name, param, tot_mag, tot_theta)

	"Generate SHG Images"
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

	"Select Data Set"
	area_sample = int(2 * (np.min((int(param['l_sample'] * conv),) + image_shg.shape[1:]) // 2))

	data_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_data'

	if not ow_data and os.path.exists(data_dir + data_file_name + '.npy'):
		print("\n Loading {} x {} pixel image data set samples {}".format(area_sample, area_sample, data_file_name))
		data_set = ut.load_npy(data_dir + data_file_name)
		dx_shg_set, dy_shg_set = ut.load_npy(data_dir + data_file_name + '_dxdy')
	else:
		data_set, indices = select_samples(image_shg, area_sample, param['min_sample'])
		print("\n Saving {} x {} pixel image data set samples {}".format(area_sample, area_sample, data_file_name))

		ut.save_npy(data_dir + data_file_name, data_set)

		pad = area_sample // 2
		dx_shg_set = np.zeros((param['min_sample'], n_frame, area_sample, area_sample))
		dy_shg_set = np.zeros((param['min_sample'], n_frame, area_sample, area_sample))

		for n in range(param['min_sample']):
			dx_shg_set[n] = dx_shg[:, indices[n][1]-pad: indices[n][1]+pad, 
						  	indices[n][0]-pad: indices[n][0]+pad]
			dy_shg_set[n] = dy_shg[:, indices[n][1]-pad: indices[n][1]+pad, 
						  		indices[n][0]-pad: indices[n][0]+pad]

		dx_shg_set = dx_shg_set.reshape(data_set.shape)
		dy_shg_set = dy_shg_set.reshape(data_set.shape)

		ut.save_npy(data_dir + data_file_name + '_dxdy', np.array((dx_shg_set, dy_shg_set)))

	"Perform Nematic Tensor Analysis"
	n_tensor = form_nematic_tensor(dx_shg_set, dy_shg_set)

	"Sample average orientational anisotopy"
	q  = nematic_tensor_analysis(n_tensor)

	print_anis_results(fig_dir, fig_name, q)

	anis_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_anis'
	print(" Saving anisotropy file {}".format(anis_file_name))
	ut.save_npy(data_dir + anis_file_name, q)

	"Smart search average orietational anisotropy"
	q = smart_nematic_tensor_analysis(n_tensor)

	print_anis_results(fig_dir, fig_name + '_smart', q)

	"Perform Fourier Analysis"
	angles, fourier_spec, sdi = fourier_transform_analysis(data_set)
	#angles = angles[len(angles)//2:]
	#fourier_spec = 2 * fourier_spec[len(fourier_spec)//2:]
	print_fourier_results(fig_dir, fig_name, angles, fourier_spec, sdi)

	"Make Gif of SHG Images"
	if ow_shg or mk_gif:
		print('\n Making Simulation SHG Gif {}/{}.gif'.format(fig_dir, fig_name))
		make_gif(fig_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, param, cell_dim * param['l_conv'], 'SHG')

	#print(' Making Simulation MD Gif {}/{}.gif'.format(fig_dir, fig_name))
	#make_gif(fig_name + '_MD', fig_dir, gif_dir, n_image, image_md * param['l_conv'], param, cell_dim * param['l_conv'], 'MD')

