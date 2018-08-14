"""
ColECM: Collagen ExtraCellular Matrix Simulation
EXPERIMENTAL ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 10/08/2018

Last Modified: 10/08/2018
"""

import numpy as np
import scipy as sp
from scipy.ndimage import filters, imread
from skimage import io, img_as_float, exposure

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os

import utilities as ut
from analysis import (fourier_transform_analysis, print_fourier_results, select_samples,
						form_nematic_tensor, nematic_tensor_analysis, print_anis_results,
						derivatives)


def experimental_image(current_dir, input_file_name):

	if input_file_name.find('display') != -1: return 

	print(input_file_name)

	cmap = 'viridis'
	size = 10

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	image_name = input_file_name
	image = imread(image_name)
	image_shg_orig = img_as_float(image)

	if image_shg_orig.ndim > 2: 
		image_shg_orig = np.sum(image_shg_orig, axis=0)
		image = np.sum(image, axis=0)

	image_shg = image_shg_orig
	#image_shg = filters.gaussian(image_shg_orig, sigma=sigma)
	image_shg = exposure.equalize_adapthist(image_shg, clip_limit=0.03)

	figure_name = ut.check_file_name(image_name, extension='tif')

	fig = plt.figure()
	plt.imshow(image_shg, cmap=cmap, interpolation='nearest', origin='lower')
	plt.savefig('{}{}.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	derivative = derivatives(image_shg)
	n_tensor = form_nematic_tensor(derivative[0].reshape((1,) + image.shape),
								   derivative[1].reshape((1,) + image.shape),
								   size=size)

	"""
	for n in range(2):
		for m in range(2):
			fig = plt.figure()
			plt.imshow(n_tensor[0,:,:,n,m], cmap='plasma', interpolation='nearest', origin='lower')
			plt.savefig('{}{}_n_tensor_{}.png'.format(fig_dir, data_file_name, n), bbox_inches='tight')
			plt.close()
	#"""

	tot_q, tot_angle = nematic_tensor_analysis(n_tensor)
	q_filtered = np.where(image_shg / image_shg.max() >= 0.2, tot_q, -1)
	angle_filtered = np.where(image_shg / image_shg.max() >= 0.2, tot_angle, -360)

	plt.figure()
	plt.imshow(q_filtered[0], cmap='binary_r', interpolation='nearest', origin='lower', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(angle_filtered[0], cmap='nipy_spectral', interpolation='nearest', origin='lower', vmin=-45, vmax=45)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Angular Histogram')
	plt.hist(angle_filtered[0].reshape(image.shape[0]*image.shape[1]), bins=100, density=True, label=figure_name, range=[-45, 45])
	plt.xlabel(r'Angle')
	plt.xlim(-45, 45)
	plt.legend()
	plt.savefig('{}{}_tot_angle_hist.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	#angles, fourier_spec, sdi = fourier_transform_analysis(image_shg.reshape((1,) + image_shg.shape))
	#angles = angles[len(angles)//2:]
	#fourier_spec = 2 * fourier_spec[len(fourier_spec)//2:]
	#print_fourier_results(fig_dir, data_file_name, angles, fourier_spec, sdi)