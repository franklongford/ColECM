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
from scipy.ndimage import filters

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


def print_anis_results(fig_dir, fig_name, tot_q, tot_angle):

	nframe = tot_q.shape[0]
	nxy = tot_q.shape[1]
	print('\n Mean anistoropy = {:>6.4f}'.format(np.mean(tot_q)))

	plt.figure()
	plt.imshow(tot_q[0], cmap='binary_r', interpolation='nearest', origin='lower', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_angle[0], cmap='nipy_spectral', interpolation='nearest', origin='lower', vmin=-45, vmax=45)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	q_hist = np.zeros(100)
	angle_hist = np.zeros(100)

	for frame in range(nframe):
		q_hist += np.histogram(tot_q[frame].flatten(), bins=100, density=True, range=[0, 1])[0] / nframe
		angle_hist += np.histogram(tot_angle[frame].flatten(), bins=100, density=True, range=[-45, 45])[0] / nframe

	plt.figure()
	plt.title('Anisotropy Histogram')
	plt.plot(np.linspace(0, 1, 100), q_hist, label=fig_name)
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig('{}{}_tot_aniso_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Angular Histogram')
	plt.plot(np.linspace(-45, 45, 100), angle_hist, label=fig_name)
	plt.xlabel(r'Angle')
	plt.xlim(-45, 45)
	plt.legend()
	plt.savefig('{}{}_tot_angle_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()


def print_fourier_results(fig_dir, fig_name, angles, fourier_spec, sdi):

	print('\n Modal Fourier Amplitude  = {:>6.4f}'.format(angles[np.argmax(fourier_spec)]))
	print(' Fourier Amplitudes Range   = {:>6.4f}'.format(np.max(fourier_spec)-np.min(fourier_spec)))
	print(' Fourier Amplitudes Std Dev = {:>6.4f}'.format(np.std(fourier_spec)))
	print(' Fourier SDI = {:>6.4f}'.format(sdi))

	print(' Creating Fouier Angle Spectrum figure {}{}_fourier.png'.format(fig_dir, fig_name))
	plt.figure(11)
	plt.title('Fourier Angle Spectrum')
	plt.plot(angles, fourier_spec, label=fig_name)
	plt.xlabel(r'Angle (deg)')
	plt.ylabel(r'Amplitude')
	plt.xlim(-180, 180)
	plt.ylim(0, 0.05)
	plt.legend()
	plt.savefig('{}{}_fourier.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def create_image(pos, sigma, n_xyz):
	"""
	create_image(pos, sigma)

	Create Gaussian convoluted image from a set of bead positions

	Parameter
	---------

	pos:  array_like (float), shape=(n_bead)
		Bead positions along n_dim dimension

	sigma:  float
		Standard deviation of Gaussian distribution

	n_xyz:  tuple (int); shape(n_dim)
		Number of pixels in each image dimension

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

	if n_dim == 2: histogram = histogram.T
	elif n_dim == 3: histogram = ut.reorder_array(histogram)

	from skimage import filters
	import time

	image_shg = filters.gaussian(histogram, sigma=sigma, mode='wrap')
	image_shg /= np.max(image_shg)

	return histogram, image_shg


def derivatives(image):

	derivative = np.zeros((2,) + image.shape)
	derivative[0] += np.gradient(image, axis=0)  #(ut.move_array_centre(image, np.array((1, 0))) - image)
	derivative[1] += np.gradient(image, axis=1)  #(ut.move_array_centre(image, np.array((0, 1))) - image)

	return derivative


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

	for image in range(n_image):
		sys.stdout.write(" Processing image {} out of {}\r".format(image, n_image))
		sys.stdout.flush()
		
		hist, image_shg[image] = create_image(traj[image], sigma, n_xyz)
		dx_shg[image], dy_shg[image] = derivatives(image_shg[image])

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
			#for bond in bonds:plt.plot(image[0][bond], image[1][bond], linestyle='dashed')
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


def form_nematic_tensor(dx_shg, dy_shg, sigma=None, size=None):
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

	n_vector:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)	

	"""

	nframe = dx_shg.shape[0]
	r_xy_2 = (dx_shg**2 + dy_shg**2)

	nxx = np.nan_to_num(dy_shg**2 / r_xy_2)
	nyy = np.nan_to_num(dx_shg**2 / r_xy_2)
	nxy = np.nan_to_num(-dx_shg * dy_shg / r_xy_2)

	if sigma != None:
		for frame in range(nframe):
			nxx[frame] = filters.gaussian_filter(nxx[frame], sigma=sigma)
			nyy[frame] = filters.gaussian_filter(nyy[frame], sigma=sigma)
			nxy[frame] = filters.gaussian_filter(nxy[frame], sigma=sigma)
	elif size != None:
		for frame in range(nframe):
			nxx[frame] = filters.uniform_filter(nxx[frame], size=size)
			nyy[frame] = filters.uniform_filter(nyy[frame], size=size)
			nxy[frame] = filters.uniform_filter(nxy[frame], size=size)

	n_vector = np.stack((nxx, nxy, nxy, nyy), -1).reshape(nxx.shape + (2,2))

	return n_vector


def select_samples(full_set, area, n_sample):
	"""
	select_samples(full_set, area, n_sample)

	Selects n_sample random sections of image stack full_set

	Parameters
	----------

	full_set:  array_like (float); shape(n_frame, n_y, n_x)
		Full set of n_frame images

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	data_set:  array_like (float); shape=(n_sample, 2, n_y, n_x)
		Sampled areas

	indices:  array_like (float); shape=(n_sample, 2)
		Starting points for random selection of full_set

	"""
	
	if full_set.ndim == 2: full_set = full_set.reshape((1,) + full_set.shape)

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

	eig_val, eig_vec = np.linalg.eig(nem_vector)
	size = eig_val.shape[:-1] + (1,)
	tot_q = eig_val.max(axis=-1) - eig_val.min(axis=-1)
	tot_angle = np.arcsin(eig_vec[:, :, :, 0, 1]) / np.pi * 180

	return tot_q, tot_angle


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

	fft_angle = np.angle(image_fft, deg=True)
	fft_freqs = np.fft.fftfreq(image_fft.size)
	angles = np.unique(fft_angle)
	fourier_spec = np.zeros(angles.shape)
	
	n_bins = fourier_spec.size

	for n in range(n_sample):
		image_fft = np.fft.fft2(image_shg[n])
		image_fft[0][0] = 0
		average_fft += np.fft.fftshift(image_fft) / n_sample	

	for i in range(n_bins):
		indices = np.where(fft_angle == angles[i])
		fourier_spec[i] += np.sum(np.abs(average_fft[indices])) / 360

	#A = np.sqrt(average_fft * fft_angle.size * fft_freqs**2 * (np.cos(fft_angle)**2 + np.sin(fft_angle)**2))

	sdi = np.mean(fourier_spec) / np.max(fourier_spec)

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

		print(dx_shg.shape)

		for n in range(param['min_sample']):
			dx_shg_set[n] = dx_shg[:, indices[n][1]-pad: indices[n][1]+pad, 
						  	indices[n][0]-pad: indices[n][0]+pad]
			dy_shg_set[n] = dy_shg[:, indices[n][1]-pad: indices[n][1]+pad, 
						  		indices[n][0]-pad: indices[n][0]+pad]

		dx_shg_set = dx_shg_set.reshape(data_set.shape)
		dy_shg_set = dy_shg_set.reshape(data_set.shape)

		ut.save_npy(data_dir + data_file_name + '_dxdy', np.array((dx_shg_set, dy_shg_set)))


	"Perform Nematic Tensor Analysis"
	n_tensor = form_nematic_tensor(dx_shg_set, dy_shg_set, size=5)
	tot_q, tot_angle = nematic_tensor_analysis(n_tensor)

	q_filtered = np.where(data_set >= 0.2, tot_q, -1)
	angle_filtered = np.where(data_set >= 0.2, tot_angle, -360)
	
	print_anis_results(fig_dir, fig_name, q_filtered, angle_filtered)

	"""
	anis_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_anis'
	print(" Saving anisotropy file {}".format(anis_file_name))
	ut.save_npy(data_dir + anis_file_name, q)

	"Smart search average orietational anisotropy"
	q = smart_nematic_tensor_analysis(n_tensor)

	print_anis_results(fig_dir, fig_name + '_smart', q)
	"""

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

