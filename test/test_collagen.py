"""
COLLAGEN FIBRE SIMULATION TEST SUITE

Created by: Frank Longford
Created on: 13/03/2018

Last Modified: 13/03/2018
"""

import numpy as np
import random

import sys, os, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/2D/')))

import utilities_2D as ut
import simulation_2D as sim


THRESH = 1E-8
cell_dim = np.array([60, 60])

pos = np.array([[20.3155606, 29.0287238],
		[20.3657056, 28.0910350],
		[19.7335474, 29.3759130]])
bond_matrix = np.array([[0, 1, 0],
			[1, 0, 1],
			[0, 1, 0]])


def test_unit_vector():

	vector = np.array([-3, 2, 6])
	answer = np.array([-0.42857143,  0.28571429,  0.85714286])
	u_vector = ut.unit_vector(vector)

	assert np.sum(u_vector - answer) <= THRESH

	vector_array = np.array([[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]])

	u_vector_array = ut.unit_vector(vector_array)

	assert np.array(vector_array).shape == u_vector_array.shape


def test_remove():

	array_1 = np.arange(50)
	array_2 = array_1 + 20
	answer = np.arange(20)

	edit_array = ut.numpy_remove(array_1, array_2)

	assert np.sum(answer - edit_array) <= THRESH


def test_param_file():

	param_file_name = 'test_param_file'
	if not os.path.exists('{}.pkl'.format(param_file_name)): ut.make_param_file(param_file_name)
	param_file = ut.read_param_file(param_file_name)

	param_file = ut.update_param_file(param_file_name, 'M', [16.0, 1.008, 1.008, 0])

	assert len(param_file['M']) == 4
	assert np.sum(np.array(param_file['M']) - 18.016) <= THRESH

	os.remove(param_file_name + '.pkl')


def test_load_save():

	test_data = np.arange(50)
	test_name = 'test_load_save'

	ut.save_npy(test_name, test_data)
	load_data = ut.load_npy(test_name)

	assert abs(np.sum(test_data - load_data)) <= THRESH

	new_test_data = test_data[:10]
	load_data = ut.load_npy(test_name, frames=range(10))

	assert abs(np.sum(new_test_data - load_data)) <= THRESH

	os.remove(test_name + '.npy')


def test_get_dxyz():

	dx_check = np.array([[0, 20.3155606 - 20.3657056, 20.3155606 - 19.7335474],
			     [20.3657056 - 20.3155606, 0, 20.3657056 - 19.7335474],
			     [19.7335474 - 20.3155606, 19.7335474 - 20.3657056, 0]])
	dy_check = np.array([[0, 29.0287238 - 28.0910350, 29.0287238 - 29.3759130],
			     [28.0910350 - 29.0287238, 0, 28.0910350 - 29.3759130],
			     [29.3759130 - 29.0287238, 29.3759130 - 28.0910350, 0]])

	dx, dy = sim.get_dx_dy(pos, cell_dim)

	assert abs(np.sum(dx - dx_check)) <= THRESH
	assert abs(np.sum(dy - dy_check)) <= THRESH


	"""
	cell_dim = np.array([60, 60, 60])

	pos = np.array([[20.3155606, 29.0287238, 58.6756206],
			[20.3657056, 28.0910350, 58.8612466],
			[19.7335474, 29.3759130, 59.3516029]])

	dx_check = np.array([[0, 20.3155606 - 20.3657056, 20.3155606 - 19.7335474],
			     [20.3657056 - 20.3155606, 0, 20.3657056 - 19.7335474],
			     [19.7335474 - 20.3155606, 19.7335474 - 20.3657056, 0]])
	dy_check = np.array([[0, 29.0287238 - 28.0910350, 29.0287238 - 29.3759130],
			     [28.0910350 - 29.0287238, 0, 28.0910350 - 29.3759130],
			     [29.3759130 - 29.0287238, 29.3759130 - 28.0910350, 0]])
	dz_check = np.array([[0, 58.6756206 - 58.8612466, 58.6756206 - 59.3516029],
			     [58.8612466 - 58.6756206, 0, 58.8612466 - 59.3516029],
			     [59.3516029 - 58.6756206, 59.3516029 - 58.8612466, 0]])

	dx, dy, dz = sim.get_dxyz(pos, cell_dim)

	assert abs(np.sum(dx - dx_check)) <= THRESH
	assert abs(np.sum(dy - dy_check)) <= THRESH
	assert abs(np.sum(dz - dz_check)) <= THRESH
	"""

def test_cos_sin_theta():

	dx, dy = sim.get_dx_dy(pos, cell_dim)
	bond_beads, dxdy_index, r_index = sim.update_bond_lists(bond_matrix)
	indices_dxy = ut.create_index(dxdy_index)

	vector = np.stack((dx[indices_dxy], dy[indices_dxy]), axis=1)
	n_vector = int(vector.shape[0])

	"Find |rij| values for each vector"
	r_vector = np.sqrt(np.sum(vector**2, axis=1))

	cos_the, sin_the, _ = sim.cos_theta(vector, r_vector)
	cos = np.arccos(cos_the)

	assert abs(np.sum(cos_the - 0.91957468)) <= THRESH
	assert abs(np.sum(sin_the - np.sin(cos))) <= THRESH
	
	

