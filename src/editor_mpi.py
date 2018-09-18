"""
ColECM: Collagen ExtraCellular Matrix Simulation
FILE EDITOR ROUTINE 

Created by: Frank Longford
Created on: 15/05/2018

Last Modified: 15/05/2018
"""

import numpy as np
import sys, os, pickle
from mpi4py import MPI

import utilities as ut
import setup


def editor_mpi(current_dir, comm, input_file_name=False, size=1, rank=0):	

	from editor import repeat_pos_array, check_edit_param

	sim_dir = current_dir + '/sim/'

	if rank == 0:
		print("\n Entering Editor\n")
		file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name)
		print("\n Loading restart file {}.npy\n".format(sim_dir + file_names['restart_file_name']))
		restart = ut.load_npy(sim_dir + file_names['restart_file_name'])
		pos = restart[0]
		vel = restart[1]
		cell_dim = pos[-1]
		pos = pos[:-1]

		if ('-nrepx' in sys.argv): n_rep_x = int(sys.argv[sys.argv.index('-nrepx') + 1]) + 1
		else: n_rep_x = 1
		if ('-nrepy' in sys.argv): n_rep_y = int(sys.argv[sys.argv.index('-nrepy') + 1]) + 1
		else: n_rep_y = 1
		if ('-nrepz' in sys.argv): n_rep_z = int(sys.argv[sys.argv.index('-nrepz') + 1]) + 1
		else: n_rep_z = 1

		param, keys = check_edit_param(sys.argv, param)

		if (n_rep_x * n_rep_y * n_rep_z) > 1:
			run_temp = True
			keys += ['n_fibril_x', 'n_fibril_y', 'n_fibril_z', 'n_fibril', 'n_bead'] 
			pos, vel, cell_dim, param = repeat_pos_array(pos, vel, cell_dim, param, n_rep_x, n_rep_y, n_rep_z)
		else: run_temp = False

		print(" New Simulation Parameters:")
		for key in keys: print(" {:<15s} : {}".format(key, param[key]))

	else:
		file_names = None
		pos = None
		cell_dim = None
		param = None
		run_temp = None

	file_names = comm.bcast(file_names, root=0)
	pos = comm.bcast(pos, root=0)
	cell_dim = comm.bcast(cell_dim, root=0)
	param = comm.bcast(param, root=0)
	run_temp = comm.bcast(run_temp, root=0)

	if run_temp:
		from simulation_mpi import equilibrate_temperature_mpi
		pos, vel = equilibrate_temperature_mpi(sim_dir, pos, cell_dim, param['bond_matrix'], param['vdw_matrix'], param, comm, size, rank)

	if rank == 0:
		print("\n Saving parameter file {}".format(file_names['param_file_name']))
		pickle.dump(param, open(sim_dir + file_names['param_file_name'] + '.pkl', 'wb'))

		print(" Saving restart file {}\n".format(file_names['restart_file_name']))
		ut.save_npy(sim_dir + file_names['restart_file_name'], (np.vstack((pos, cell_dim)), vel))


