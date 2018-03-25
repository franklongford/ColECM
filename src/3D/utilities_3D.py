"""
COLLAGEN FIBRE SIMULATION 3D

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 09/03/2018
"""

import numpy as np
import scipy as sp
import random

import sys, os, pickle


SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)
SQRTPI = np.sqrt(np.pi)


def get_dx_dy_dz(pos, N, cell_dim):
	"""
	get_dx_dy_dz(pos, N, cell_dim)

	Calculate distance vector between two beads
	"""

	N = pos.shape[0]
	temp_pos = np.moveaxis(pos, 0, 1)

	dx = np.tile(temp_pos[0], (N, 1))
	dy = np.tile(temp_pos[1], (N, 1))
	dz = np.tile(temp_pos[2], (N, 1))

	dx = dx.T - dx
	dy = dy.T - dy
	dz = dz.T - dz

	dx -= cell_dim[0] * np.array(2 * dx / cell_dim[0], dtype=int)
	dy -= cell_dim[1] * np.array(2 * dy / cell_dim[1], dtype=int)
	dz -= cell_dim[2] * np.array(2 * dz / cell_dim[2], dtype=int)

	return dx, dy, dz


def create_index(array):
	"""
	create_index(array)

	Takes a list of 3D indicies and returns an index array that can be used to access elements in a 3D numpy array
	"""
    
	return (np.array(array.T[0]), np.array(array.T[1]), np.array(array.T[2]))

