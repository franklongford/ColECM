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
from analysis import print_thermo_results, print_vector_results, print_anis_results, print_fourier_results, form_nematic_tensor, make_gif, select_samples
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
	bond_indices = np.nonzero(np.triu(param['bond_matrix']))
	n_bond = bond_indices[0].shape[0]
	bond_list = np.zeros((param['n_dim'], n_bond))

	tot_mag = np.zeros((n_image, param['n_fibril']))
	#tot_theta = np.zeros((n_image, param['n_fibril']))

	#tot_mag = np.zeros((n_image, n_bond))
	tot_theta = np.zeros((n_image, n_bond))

	for image in range(rank, n_image, size):
		pos = traj[image]

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

	tot_mag = np.sum(comm.gather(tot_mag), axis=0)
	tot_theta = np.sum(comm.gather(tot_theta), axis=0)

	return tot_theta, tot_mag


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

	for image in range(rank, n_image, size)
		sys.stdout.write(" Processing image {} out of {}\r".format(image, n_image))
		sys.stdout.flush()
		
		hist, image_shg[image] = create_image(traj[image], sigma, n_xyz)
		
	image_shg = comm.allreduce(image_shg, op=MPI.SUM)

	return image_shg


def analysis_mpi(current_dir, comm, input_file_name=False, size=1, rank=0):


	sim_dir = current_dir + '/sim/'
	gif_dir = current_dir + '/gif/'
	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	ow_shg = ('-ow_shg' in sys.argv)
	ow_data = ('-ow_data' in sys.argv)
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
	
	if not ow_shg:
		if rank == 0:
			try: image_shg = ut.load_npy(sim_dir + image_file_name, range(0, n_frame, param['skip']))	
			except: ow_shg = True
		else: image_shg = None
		ow_shg = comm.bcast(ow_shg, root=0)
	
	if ow_shg:
		"Generate Gaussian convoluted images and intensity derivatives"
		image_shg = shg_images_mpi(image_md, param['vdw_sigma'] * conv, n_xyz, 2 * param['rc'] * conv, comm, size, rank)
		if rank == 0:
			print("\n Saving image files {}".format(file_names['output_file_name']))
			ut.save_npy(sim_dir + image_file_name, image_shg)
		image_shg = np.array([image_shg[i] for i in range(0, n_frame, param['skip'])])
	else: image_shg = comm.bcast(image_shg, root=0)

	fig_name += '_{}_{}'.format(param['res'], param['sharp'])
	
	if rank == 0: 
		"Make Gif of SHG Images"
		if ow_shg or mk_gif:
			print('\n Making Simulation SHG Gif {}/{}.gif'.format(fig_dir, fig_name))
			make_gif(fig_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, param, cell_dim * param['l_conv'], 'SHG')

	#print(' Making Simulation MD Gif {}/{}.gif'.format(fig_dir, fig_name))
	#make_gif(fig_name + '_MD', fig_dir, gif_dir, n_image, image_md * param['l_conv'], param, cell_dim * param['l_conv'], 'MD')

