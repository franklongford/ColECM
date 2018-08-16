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

	return image_shg



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
	
	if not ow_shg:
		try: image_shg = ut.load_npy(sim_dir + image_file_name, range(0, n_frame, param['skip']))	
		except: ow_shg = True
	
	if ow_shg:
		"Generate Gaussian convoluted images and intensity derivatives"
		image_shg = shg_images(image_md, param['vdw_sigma'] * conv, n_xyz, 2 * param['rc'] * conv)

		print("\n Saving image files {}".format(file_names['output_file_name']))
		ut.save_npy(sim_dir + image_file_name, image_shg)

		image_shg = np.array([image_shg[i] for i in range(0, n_frame, param['skip'])])

	fig_name += '_{}_{}'.format(param['res'], param['sharp'])

	"Make Gif of SHG Images"
	if ow_shg or mk_gif:
		print('\n Making Simulation SHG Gif {}/{}.gif'.format(fig_dir, fig_name))
		make_gif(fig_name + '_SHG', fig_dir, gif_dir, n_image, image_shg, param, cell_dim * param['l_conv'], 'SHG')

		#print(' Making Simulation MD Gif {}/{}.gif'.format(fig_dir, fig_name))
		#make_gif(fig_name + '_MD', fig_dir, gif_dir, n_image, image_md * param['l_conv'], param, cell_dim * param['l_conv'], 'MD')

