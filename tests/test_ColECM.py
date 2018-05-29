"""
COLLAGEN FIBRIL SIMULATION TEST SUITE

Created by: Frank Longford
Created on: 13/03/2018

Last Modified: 13/03/2018
"""

import numpy as np
import sys, os, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import utilities as ut
import simulation as sim
import setup
import sim_tools_2D as sim_2D
import sim_tools_3D as sim_3D


THRESH = 1E-7
cell_dim_2D = np.array([60, 60])
cell_dim_3D = np.array([60, 60, 60])

pos_2D = np.array([[20.3155606, 29.0287238],
		   [20.3657056, 28.0910350],
		   [19.7335474, 29.3759130]])

pos_3D = np.array([[20.3155606, 29.0287238, 58.6756206],
	           [20.3657056, 28.0910350, 58.8612466],
		   [19.7335474, 29.3759130, 59.3516029]])

bond_matrix = np.array([[0, 1, 0],
			[1, 0, 1],
			[0, 1, 0]])

vdw_matrix = np.array([[0, 8, 1],
		       [8, 0, 8],
		       [1, 8, 0]])

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

	param_file_name = 'test_param_file_param.pkl'
	param_file_name = ut.check_file_name(param_file_name, 'param', 'pkl')

	assert param_file_name == 'test_param_file'

	param_file_name = 'test_param_file.pkl'
	param_file_name = ut.check_file_name(param_file_name, extension='pkl')

	assert param_file_name == 'test_param_file'

	param_file_name = 'test_param_file_param'
	param_file_name = ut.check_file_name(param_file_name, file_type='param')

	assert param_file_name == 'test_param_file'

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

	dx, dy = ut.get_distances(pos_2D, cell_dim_2D)

	assert abs(np.sum(dx - dx_check)) <= THRESH
	assert abs(np.sum(dy - dy_check)) <= THRESH


	dx_check = np.array([[0, 20.3155606 - 20.3657056, 20.3155606 - 19.7335474],
			     [20.3657056 - 20.3155606, 0, 20.3657056 - 19.7335474],
			     [19.7335474 - 20.3155606, 19.7335474 - 20.3657056, 0]])
	dy_check = np.array([[0, 29.0287238 - 28.0910350, 29.0287238 - 29.3759130],
			     [28.0910350 - 29.0287238, 0, 28.0910350 - 29.3759130],
			     [29.3759130 - 29.0287238, 29.3759130 - 28.0910350, 0]])
	dz_check = np.array([[0, 58.6756206 - 58.8612466, 58.6756206 - 59.3516029],
			     [58.8612466 - 58.6756206, 0, 58.8612466 - 59.3516029],
			     [59.3516029 - 58.6756206, 59.3516029 - 58.8612466, 0]])

	dx, dy, dz = ut.get_distances(pos_3D, cell_dim_3D)

	assert abs(np.sum(dx - dx_check)) <= THRESH
	assert abs(np.sum(dy - dy_check)) <= THRESH
	assert abs(np.sum(dz - dz_check)) <= THRESH
	

def test_cos_sin_theta():

	dx, dy = ut.get_distances(pos_2D, cell_dim_2D)
	bond_beads, dxdy_index, r_index = ut.update_bond_lists(bond_matrix)
	indices_dxy = ut.create_index(dxdy_index)

	vector = np.stack((dx[indices_dxy], dy[indices_dxy]), axis=1)
	n_vector = int(vector.shape[0])

	"Find |rij| values for each vector"
	r_vector = np.sqrt(np.sum(vector**2, axis=1))

	cos_the, sin_the, _ = sim_2D.cos_sin_theta(vector, r_vector)
	check_sin_the = np.array([-0.39291528])

	assert abs(np.sum(cos_the - 0.91957468)) <= THRESH
	assert abs(np.sum(sin_the - check_sin_the)) <= THRESH

	dx, dy, dz = ut.get_distances(pos_3D, cell_dim_3D)
	bond_beads, dxdydz_index, r_index = ut.update_bond_lists(bond_matrix)
	indices_dxyz = ut.create_index(dxdy_index)

	vector = np.stack((dx[indices_dxyz], dy[indices_dxyz], dz[indices_dxyz]), axis=1)
	n_vector = int(vector.shape[0])

	"Find |rij| values for each vector"
	r_vector = np.sqrt(np.sum(vector**2, axis=1))

	cos_the, sin_the, _ = sim_3D.cos_sin_theta(vector, r_vector)
	check_sin_the = np.array([[-0.48198494, -0.09796533, -0.36466797]])

	assert abs(np.sum(cos_the - 0.79063936)) <= THRESH
	assert abs(np.sum(sin_the - check_sin_the)) <= THRESH
	

def test_pot_energy_frc():

	param = setup.get_param_defaults()
	param['n_fibril_x'] = 1
	param['n_fibril_y'] = 1
	param['n_fibril_z'] = 1
	param['l_fibril'] = 3

	distances = ut.get_distances(pos_2D, cell_dim_2D)
	bond_beads, dxdy_index, r_index = ut.update_bond_lists(bond_matrix)
	r2 = np.sum(distances**2, axis=0)
	verlet_list = ut.check_cutoff(r2, param['rc']**2)
	
	pot_energy, new_frc, _ = sim_2D.calc_energy_forces(distances, r2, param, bond_matrix, vdw_matrix, 
					verlet_list, bond_beads, dxdy_index, r_index)

	check_frc = np.array([[ 12277.59052347,  -6225.74404829],
 			      [ 41.48708925,  -1095.43380772],
 			      [-12319.07761272,   7321.17785601]])

	assert abs(pot_energy - 423.5238450131211) <= THRESH
	assert abs(np.sum(new_frc - check_frc)) <= THRESH

	param['n_dim'] = 3
	distances = ut.get_distances(pos_3D, cell_dim_3D)
	bond_beads, dxdydz_index, r_index = ut.update_bond_lists(bond_matrix)
	r2 = np.sum(distances**2, axis=0)
	verlet_list = ut.check_cutoff(r2, param['rc']**2)
	
	pot_energy, new_frc, _ = sim_3D.calc_energy_forces(distances, r2, param, bond_matrix, vdw_matrix, 
					verlet_list, bond_beads, dxdy_index, r_index)

	check_frc = np.array([[  23.97941507,  770.44703468, -238.8615784 ],
 			      [  27.2045763,  -777.5797729,   172.30544761],
			      [ -51.18399136,    7.13273822,   66.55613079]])

	assert abs(pot_energy - 31.332909531802844) <= THRESH
	assert abs(np.sum(new_frc - check_frc)) <= THRESH


def test_grow_fibril():

	param = setup.get_param_defaults()
	n_attempts = 15
	param['l_fibril'] = 20
	param['n_bead'] = param['n_fibril'] * param['l_fibril'] 

	for i in range(n_attempts): setup.create_pos_array(param)

	param['n_dim'] = 3

	for i in range(n_attempts): setup.create_pos_array(param)

def test_mpi():

	from mpi4py import MPI

	param = setup.get_param_defaults()
	param['l_fibril'] = 10
	param['n_bead'] = param['n_fibril'] * param['l_fibril'] 

	if rank == 0: pos, cell_dim, param = setup.create_pos_array(param)

	

