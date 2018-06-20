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
from mpi4py import MPI

import utilities as ut
from analysis import print_thermo_results, print_vector_results, print_anis_results, print_fourier_results, form_nematic_tensor, make_gif
import setup

SQRT2 = np.sqrt(2)
SQRTPI = np.sqrt(np.pi)


def fibre_vector_analysis_mpi(traj, cell_dim, param, comm, size, rank):
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

	for image in range(rank, n_image, size):
		pos = traj[image]

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

	tot_mag = np.sum(comm.gather(tot_mag), axis=0)
	tot_theta = np.sum(comm.gather(tot_theta), axis=0)

	return tot_theta, tot_mag


def create_image_mpi(pos, std, n_xyz, r, comm, size, rank):
	"""
	create_image_mpi(pos_x, pos_y, sigma, n_xyz, r)

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

	for i in range(rank, indices.shape[0], size):
		index = indices[i]

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

	image = comm.allreduce(image, op=MPI.SUM)
	image = image.T
	
	return histogram, image


def fibril_align_mpi(histogram, std, n_xyz, dxdydz, r, non_zero, comm, size, rank):
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

	for i in range(rank, indices.shape[0], size):
		index = indices[i]
	
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


	dx_grid = comm.allreduce(dx_grid, op=MPI.SUM)
	dy_grid = comm.allreduce(dy_grid, op=MPI.SUM)

	dx_grid = dx_grid.T
	dy_grid = dy_grid.T

	return dx_grid, dy_grid


def shg_images_mpi(traj, sigma, n_xyz, cut, comm, size, rank):
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
		
		hist, image_shg[image] = create_image_mpi(traj[image], sigma, n_xyz, r, comm, size, rank)
		dx_shg[image], dy_shg[image] = fibril_align_mpi(hist, sigma, n_xyz, dxdydz, r, non_zero, comm, size, rank)

	return image_shg, dx_shg, dy_shg


def nematic_tensor_analysis_mpi(n_vector, area, min_sample, comm, size, rank, thresh = 0.05):
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

	tot_q = np.zeros(n_frame)
	av_q = []

	pad = int(area / 2 - 1)

	analysing = True
	sample = size

	while analysing:

		av_eigval = np.zeros((n_frame, 2))
		av_eigvec = np.zeros((n_frame, 2, 2))

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad

		cut_n_vector = n_vector[:, :, start_y-pad: start_y+pad, 
					      start_x-pad: start_x+pad]

		av_n = np.reshape(np.mean(cut_n_vector, axis=(2, 3)), (n_frame, 2, 2))

		for frame in range(n_frame):

			eig_val, eig_vec = np.linalg.eigh(av_n[frame])

			av_eigval[frame] = eig_val
			av_eigvec[frame] = eig_vec

		tot_q += (av_eigval.T[1] - av_eigval.T[0])

		gather_q = comm.gather(np.mean(tot_q) / sample)

		if rank == 0:
			av_q += gather_q
			if sample >= min_sample:
				q_mov_av = ut.cum_mov_average(av_q)
				analysing = (q_mov_av[-1] - q_mov_av[-2]) > thresh

		analysing = comm.bcast(analysing, root=0)

		sample += size

	tot_q = comm.allreduce(tot_q, op=MPI.SUM)

	return tot_q / sample, sample


def fourier_transform_analysis_mpi(image_shg, area, n_sample, comm, size, rank):
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

	for n in range(rank, n_sample, size):

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

	average_fft = comm.allreduce(average_fft, op=MPI.SUM)

	for i in range(n_bins):
		indices = np.where(fft_angle == angles[i])
		fourier_spec[i] += np.sum(np.abs(average_fft[indices])) / 360

	return angles, fourier_spec


def analysis(current_dir, comm, input_file_name=False, size=1, rank=0):


	sim_dir = current_dir + '/sim/'
	gif_dir = current_dir + '/gif'
	fig_dir = current_dir + '/fig'

	ow_shg = ('-ow_shg' in sys.argv)
	mk_gif = ('-mk_gif' in sys.argv)

	if rank == 0:
		print("\n " + " " * 15 + "----Beginning Image Analysis----\n")
		if not os.path.exists(gif_dir): os.mkdir(gif_dir)
		if not os.path.exists(fig_dir): os.mkdir(fig_dir)

		file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name)
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

	else:
		file_names = None
		param = None
		tot_pos = None

	file_names = comm.bcast(file_names, root=0)
	param = comm.bcast(param, root=0)
	tot_pos = comm.bcast(tot_pos, root=0)
	fig_name = file_names['gif_file_name'].split('/')[-1]
	
	n_frame = tot_pos.shape[0]
	n_image = int(n_frame / param['skip'])
	cell_dim = tot_pos[0][-1]
	n_xyz = tuple(np.array(cell_dim * param['l_conv'] * param['res'], dtype=int))
	conv = param['l_conv'] / param['sharp'] * param['res']

	image_md = np.moveaxis([tot_pos[n][:-1] for n in range(0, n_frame)], 2, 1)

	"Perform Fibre Vector analysis"
	tot_theta, tot_mag = fibre_vector_analysis_mpi(image_md, cell_dim, param, comm, size, rank)
	if rank == 0: print_vector_results(fig_dir, fig_name, param, tot_mag, tot_theta)


	image_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_{}_{}_{}_image_shg'.format(n_frame, param['res'], param['sharp'])
	dx_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_{}_{}_{}_dx_shg'.format(n_frame, param['res'], param['sharp'])
	dy_file_name = ut.check_file_name(file_names['output_file_name'], 'out', 'npy') + '_{}_{}_{}_dy_shg'.format(n_frame, param['res'], param['sharp'])

	if not ow_shg:
		if rank == 0:
			try:
				image_shg = ut.load_npy(sim_dir + image_file_name, range(0, n_frame, param['skip']))	
				dx_shg = ut.load_npy(sim_dir + dx_file_name, range(0, n_frame, param['skip']))
				dy_shg = ut.load_npy(sim_dir + dy_file_name, range(0, n_frame, param['skip']))
			except: ow_shg = True
		else:
			image_shg = None
			dx_shg = None
			dy_shg = None

		ow_shg = comm.bcast(ow_shg, root=0)
	
	if ow_shg:
		"Generate Gaussian convoluted images and intensity derivatives"
		image_shg, dx_shg, dy_shg = shg_images_mpi(image_md, param['vdw_sigma'] * conv, n_xyz, 2 * param['rc'] * conv, comm, size, rank)

		if rank == 0:
			print("\n Saving image files {}".format(file_names['output_file_name']))
			ut.save_npy(sim_dir + image_file_name, image_shg)
			ut.save_npy(sim_dir + dx_file_name, dx_shg)
			ut.save_npy(sim_dir + dy_file_name, dy_shg)

		image_shg = np.array([image_shg[i] for i in range(0, n_frame, param['skip'])])
		dx_shg = np.array([dx_shg[i] for i in range(0, n_frame, param['skip'])])
		dy_shg = np.array([dy_shg[i] for i in range(0, n_frame, param['skip'])])
	else:

		image_shg = comm.bcast(image_shg, root=0)
		dx_shg = comm.bcast(dx_shg, root=0)
		dy_shg = comm.bcast(dy_shg, root=0)

	
	fig_name += '_{}_{}'.format(param['res'], param['sharp'])

	"Perform Nematic Tensor Analysis"

	area_sample = int(2 * (np.min((param['l_sample'],) + image_shg.shape[1:]) // 2))

	n_tensor = form_nematic_tensor(dx_shg, dy_shg)
	"Sample average orientational anisotopy"
	q, n_sample = nematic_tensor_analysis_mpi(n_tensor, area_sample, param['min_sample'], comm, size, rank)

	if rank == 0: print_anis_results(fig_dir, fig_name, q)

	"Perform Fourier Analysis"
	angles, fourier_spec = fourier_transform_analysis_mpi(image_shg, area_sample, n_sample, comm, size, rank)

	#angles = angles[len(angles)//2:]
	#fourier_spec = 2 * fourier_spec[len(fourier_spec)//2:]
	if rank == 0: 
		print_fourier_results(fig_dir, fig_name, angles, fourier_spec)

		"Make Gif of SHG Images"
		if ow_shg or mk_gif:
			print('\n Making Simulation SHG Gif {}/{}.gif'.format(fig_dir, fig_name))
			make_gif(fig_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, param, cell_dim * param['l_conv'], 'SHG')

	#print(' Making Simulation MD Gif {}/{}.gif'.format(fig_dir, fig_name))
	#make_gif(fig_name + '_MD', fig_dir, gif_dir, n_image, image_md * param['l_conv'], param, cell_dim * param['l_conv'], 'MD')

