"""
ColECM: Collagen ExtraCellular Matrix Simulation
SIMULATION ROUTINE 

Created by: Frank Longford
Created on: 13/04/2018

Last Modified: 19/04/2018
"""

import numpy as np
import sys, os, time
from mpi4py import MPI

import utilities as ut
import setup


def velocity_verlet_alg_mpi(pos, vel, frc, virial_tensor, param, pos_indices, bond_indices, frc_indices, angle_indices, 
				angle_bond_indices, vdw_coeff, virial_indicies, dt, sqrt_dt, cell_dim, calc_energy_forces,
				comm, size, rank, NPT=False):
	"""
	velocity_verlet_alg_mpi(pos, vel, frc, beta, virial_tensor, param, pos_indices, bond_indices, frc_indices, angle_indices, 
				angle_bond_indices, vdw_coeff, virial_indicies, dt, sqrt_dt, cell_dim, calc_energy_forces,
				comm, size, rank, NPT=False)

	Integrate positions and velocities through time using verlocity verlet algorithm

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	vel: array_like (float); shape=(n_bead, n_dim)
		Velocity of each bead in all collagen fibrils

	frc: array_like (float); shape=(n_bead, n_dim)
		Forces acting upon each bead in all collagen fibrils

	beta: array_like (float); shape=(n_bead, n_dim)
		Random components acting upon each bead in all collagen fibrils

	virial_tensor:  array_like (float); shape=(n_dim, n_dim)
		Virial term of pressure tensor components

	param:  dict
		Dictionary of simulation and analysis parameters
 
	pos_indices: array_like (int); shape=(n_pos, n_dim)
		List of bead indices to include in pairwise distance matrix

	bond_indices: array_like (int); shape=(n_bond, 2)
		List of pairwise indices to include in bond force calculation in local pos array

	frc_indices:  array_like, (int); shape=(n_bond, 2)
		List of pairwise indices to include in bond force calculation in global pos array

	angle_indices:  array_like, (int); shape=(n_angle, 3)
		List of triplet indices to include in angle force calculation in global pos array

	angle_bond_indices:  array_like, (int); shape=(n_bond_angle, 2)
		List of pairwise indices for bonds included in angle force calculation in global pos array

	vdw_coeff:  array_like (int); shape=(n_bead, n_pos)
		Array of coefficient weightings for vdw forces

	virial_indicies: array_like (int); shape=(n_bead, n_pos)
		List of pairwise indices to include in virial sum

	dt:  float
		Length of time step

	sqrt_dt:  float
		Square root of time step

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	NPT:  bool
		Whether to run NPT ensemble (NVT otherwise)

	
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Updated positions of n_bead beads in n_dim

	vel:  array_like (float); shape=(n_bead, n_dim)
		Updated velocity of each bead in all collagen fibrils

	frc:  array_like (float); shape=(n_pos, n_dim)
		Updated forces acting upon each bead in all collagen fibrils

	pot_energy:  float
		Total energy of simulation

	virial_tensor:  array_like (float); shape=(n_dim, n_dim)
		Virial term of pressure tensor components

	"""

	if rank == 0: beta = np.random.normal(0, 1, (param['n_bead'], param['n_dim']))
	else: beta = None
	beta = comm.bcast(beta, root=0)

	vel += frc / param['mass'] * dt

	if NPT:
		kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
		P_t = 1 / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
		mu = (1 + (param['lambda_p'] * (P_t - param['P_0'])))**(1./3) 

	d_vel = param['sigma'] * beta - param['gamma'] * vel
	pos += (vel + 0.5 * d_vel) * dt
	vel += d_vel

	if NPT:
		pos = mu * pos
		cell_dim = mu * cell_dim

	cell = np.tile(cell_dim, (param['n_bead'], 1)) 
	pos += cell * (1 - np.array((pos + cell) / cell, dtype=int))
	
	frc, pot_energy, virial_tensor = calc_energy_forces(pos, cell_dim, pos_indices, bond_indices, frc_indices, 
							angle_indices, angle_bond_indices, vdw_coeff, virial_indicies, param)

	pot_energy = np.sum(comm.gather(pot_energy, root=0))
	frc = comm.allreduce(frc, op=MPI.SUM)
	virial_tensor = comm.allreduce(virial_tensor, op=MPI.SUM)

	return pos, vel, frc, cell_dim, pot_energy, virial_tensor


def equilibrate_temperature(sim_dir, pos, cell_dim, bond_matrix, vdw_matrix, param, comm, size=1, rank=0, inc=0.1, thresh=5E-2):
	"""
	equilibrate_temperature(pos, vel, cell_dim, bond_matrix, vdw_matrix, param, thresh=2E-2)

	Equilibrate temperature of system

	Parameters
	----------
	
	pos:  array_like (float); shape=(n_bead, n_dim)
		Updated positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Updated velocity of each bead in all collagen fibrils

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	vdw_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a non-bonded interaction is present between two beads

	param:  dict
		Dictionary of simulation and analysis parameters

	thresh:  float (optional)
		Threshold difference between average system and reference temperature

		
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Updated velocity of each bead in all collagen fibrils

	"""

	if rank == 0: print("\n" + " " * 15 + "----Equilibrating Temperature----\n")

	if param['n_dim'] == 2: from sim_tools_2D import calc_energy_forces_mpi
	elif param['n_dim'] == 3: from sim_tools_3D import calc_energy_forces_mpi

	dt = param['dt'] / 2
	sqrt_dt = np.sqrt(dt)
	n_dof = param['n_dim'] * param['n_bead'] 
	vel = np.zeros(pos.shape)

	sim_state = setup.calc_state_mpi(pos, cell_dim, bond_matrix, vdw_matrix, param, comm, size, rank)
	frc, pot_energy, virial_tensor, bond_indices, angle_indices, angle_bond_indices = sim_state

	pos_indices = np.array_split(np.arange(param['n_bead']), size)[rank]
	frc_indices = (bond_indices[0] + pos_indices[0], bond_indices[1])
	vdw_coeff = np.array_split(param['vdw_matrix'], size)[rank]
	virial_indicies = ut.create_index(np.argwhere(np.array_split(np.tri(param['n_bead']).T, size)[rank]))

	kBT = 2 * ut.kin_energy(vel, param['mass'], param['n_dim']) / n_dof
	step = 1
	kBT_array = [kBT]
	optimising = True
	ref_kBT = param['kBT']
	param['kBT'] = inc
	param['sigma'] = np.sqrt(param['gamma'] * (2 - param['gamma']) * (param['kBT'] / param['mass']))

	if rank == 0:
		print(" Starting kBT:    {:>10.4f}\n Reference kBT:   {:>10.4f}\n".format(kBT, param['kBT']))
		print(" {:^18s} | {:^18s} | {:^18s}".format('Step', 'Ref kBT', 'kBT'))
		print(" " + "-" * 60)

	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)
	verlet_list_rc = ut.check_cutoff(r2, param['rc']**2)

	while optimising:
	
		"""	
		sim_state = velocity_verlet_alg(pos, vel, frc, virial_tensor, param, bond_matrix, vdw_matrix, 
			verlet_list_rc, angle_indices, angle_bond_indices, r_indices, param['dt']/2, sqrt_dt, cell_dim)
		(pos, vel, frc, cell_dim, pot_energy, virial_tensor, r2) = sim_state
		verlet_list_rc = ut.check_cutoff(r2, param['rc']**2)

		#"""
		sim_state = velocity_verlet_alg_mpi(pos, vel, frc, virial_tensor, param, pos_indices, bond_indices, frc_indices, angle_indices, 
				angle_bond_indices, vdw_coeff, virial_indicies, dt, sqrt_dt, cell_dim, calc_energy_forces_mpi, comm, size, rank)

		(pos, vel, frc, cell_dim, pot_energy, virial_tensor) = sim_state
		#"""		

		"""
		"DYNAMIC BONDS - not yet implemented fully"
		
		if step % 1 == 0: 
			param['bond_matrix'], update = ut.bond_check(param['bond_matrix'], fib_end, r2, param['rc'], param['bond_rb'], param['vdw_sigma'])
			if update:
				bond_beads, dist_index, r_index, fib_end = ut.update_bond_lists(bond_matrix)
		"""
		kBT = 2 * ut.kin_energy(vel, param['mass'], param['n_dim']) / n_dof
		kBT_array.append(kBT)

		if step % 250 == 0: 
			av_kBT = np.mean(kBT_array)
			kBT_array = [kBT]
			if rank == 0: print(" {:18d} | {:>18.4f} | {:>18.4f}".format(step, param['kBT'], av_kBT))
			if abs(av_kBT - param['kBT']) <= thresh:
				param['kBT'] += inc
				param['sigma'] = np.sqrt(param['gamma'] * (2 - param['gamma']) * (param['kBT'] / param['mass']))
				if ref_kBT <= param['kBT']: optimising = False

		step += 1

	param['kBT'] = ref_kBT
	param['sigma'] = np.sqrt(param['gamma'] * (2 - param['gamma']) * (param['kBT'] / param['mass']))

	if rank == 0:
		print("\n No. iterations:   {:>10d}".format(step))
		print(" Final kBT:   {:>10.4f}".format(kBT))

	return pos, vel


def equilibrate_density(pos, vel, cell_dim, bond_matrix, vdw_matrix, param, comm, size=1, rank=0, thresh=2E-3):
	"""
	equilibrate_density(pos, vel, cell_dim, bond_matrix, vdw_matrix, param, thresh=2E-3)

	Equilibrate density of system

	Parameters
	----------
	
	pos:  array_like (float); shape=(n_bead, n_dim)
		Updated positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Updated velocity of each bead in all collagen fibrils

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	vdw_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a non-bonded interaction is present between two beads

	param:  dict
		Dictionary of simulation and analysis parameters

	thresh:  float (optional)
		Threshold difference between average system and reference pressure

		
	Returns
	-------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	vel: array_like, dtype=float
		Updated velocity of each bead in all collagen fibrils

	cell_dim:  array_like (float); shape=(n_dim)
		Simulation cell dimensions in n_dim dimensions

	"""


	if rank == 0: print("\n" + " " * 15 + "----Equilibrating Density----\n")

	if param['n_dim'] == 2: from sim_tools_2D import calc_energy_forces_mpi
	elif param['n_dim'] == 3: from sim_tools_3D import calc_energy_forces_mpi

	sqrt_dt = np.sqrt(param['dt'])
	n_dof = param['n_dim'] * param['n_bead']

	sim_state = setup.calc_state_mpi(pos, cell_dim, bond_matrix, vdw_matrix, param, comm, size, rank)
	frc, pot_energy, virial_tensor, bond_indices, angle_indices, angle_bond_indices = sim_state

	pos_indices = np.array_split(np.arange(param['n_bead']), size)[rank]
	frc_indices = (bond_indices[0] + pos_indices[0], bond_indices[1])
	vdw_coeff = np.array_split(param['vdw_matrix'], size)[rank]
	virial_indicies = ut.create_index(np.argwhere(np.array_split(np.tri(param['n_bead']).T, size)[rank]))

	kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
	P = 1. / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
	kBT = 2 * kin_energy / n_dof
	step = 1
	kBT_array = [kBT]
	P_array = [P]

	step = 1
	optimising = True

	if rank == 0:
		print(" Starting density:      {:>10.4f}\n Reference density:     {:>10.4f}".format(param['n_bead'] / np.prod(cell_dim), param['density']))
		print(" Starting pressure:     {:>10.4f}\n Max pressure:          {:>10.4f}".format(P, param['P_0']))
		print(" Starting volume:       {:>10.4f}\n Starting temperature:  {:>10.4f}\n".format(np.prod(cell_dim), kBT))
		print(" {:^12s} | {:^12s} | {:^12s} | {:^12s} ".format('Step', 'Av Pressure', 'Av Temperature', 'Density'))
		print(" " + "-" * 56)

	while optimising:

		sim_state = velocity_verlet_alg_mpi(pos, vel, frc, virial_tensor, param, pos_indices, bond_indices, frc_indices, angle_indices, 
				angle_bond_indices, vdw_coeff, virial_indicies, param['dt'], sqrt_dt, cell_dim, calc_energy_forces_mpi, comm, size, rank, NPT=True)

		(pos, vel, frc, cell_dim, pot_energy, virial_tensor) = sim_state

		"""
		sim_state = velocity_verlet_alg(pos, vel, frc, virial_tensor, param, bond_matrix, vdw_matrix, 
			verlet_list_rc, bond_beads, dist_index, r_index, param['dt']/2, sqrt_dt, cell_dim, NPT=True)
		(pos, vel, frc, cell_dim, pot_energy, virial_tensor, r2) = sim_state
		verlet_list_rc = ut.check_cutoff(r2, param['rc']**2)
		"DYNAMIC BONDS - not yet implemented fully"
		if step % 1 == 0: 
			param['bond_matrix'], update = ut.bond_check(param['bond_matrix'], fib_end, r2, param['rc'], param['bond_rb'], param['vdw_sigma'])
			if update:
				bond_beads, dist_index, r_index, fib_end = ut.update_bond_lists(bond_matrix)
		"""
		kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
		P = 1. / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
		kBT = 2 * kin_energy / n_dof

		P_array.append(P)
		kBT_array.append(kBT)
		den = param['n_bead'] / np.prod(cell_dim)

		optimising = abs(den - param['density']) > thresh
		optimising *= den < param['density']

		if step % 2000 == 0: 
			av_P = np.mean(P_array)
			av_kBT = np.mean(kBT_array)

			P_array = [P]
			kBT_array = [kBT]

			if av_P >= param['P_0']: optimising = False
			if rank == 0: print(" {:12d} | {:>12.4f} | {:>12.4f} | {:>12.4f}".format(step, av_P, av_kBT, den))
		step += 1

	if rank == 0:
		av_P = np.mean(P_array)
		av_kBT = np.mean(kBT_array)
		print(" {:12d} | {:>12.4f} | {:>12.4f} | {:>12.4f}".format(step, av_P, av_kBT, den))
		print("\n No. iterations:   {:>10d}".format(step))
		print(" Final density:   {:>10.4f}".format(param['n_bead'] / np.prod(cell_dim)))
		print(" Final volume:     {:>10.4f}".format(np.prod(cell_dim)))

	return pos, vel, cell_dim


def simulation(current_dir, comm, input_file_name=False, size=1, rank=0):	

	sim_dir = current_dir + '/sim/'

	if rank == 0:
		print("\n " + " " * 15 + "----Entering Setup----\n")
		setup_time_start = time.time()
		if not os.path.exists(sim_dir): os.mkdir(sim_dir)
		file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name)
	else:
		file_names = None
		param = None
	file_names = comm.bcast(file_names, root=0)
	param = comm.bcast(param, root=0)


	if param['n_dim'] == 2: from sim_tools_2D import calc_energy_forces_mpi
	elif param['n_dim'] == 3: from sim_tools_3D import calc_energy_forces_mpi

	n_frames = int(param['n_step'] / param['save_step'])
	dig = len(str(param['n_step']))
	sqrt_dt = np.sqrt(param['dt'])

	pos, vel, cell_dim, param = setup.import_files_mpi(sim_dir, file_names, param, comm, size, rank)

	n_dof = param['n_dim'] * param['n_bead'] 

	if rank == 0:
		tot_pos = np.zeros((n_frames, param['n_bead'] + 1, param['n_dim']))
		tot_vel = np.zeros((n_frames, param['n_bead'], param['n_dim']))
		tot_frc = np.zeros((n_frames, param['n_bead'], param['n_dim']))

		tot_temp = np.zeros(param['n_step'])
		tot_energy = np.zeros(param['n_step'])
		tot_press = np.zeros(param['n_step'])
		tot_vol = np.zeros(param['n_step'])

	sim_state = setup.calc_state_mpi(pos, cell_dim, param['bond_matrix'], param['vdw_matrix'], param, comm, size, rank)
	frc, pot_energy, virial_tensor, bond_indices, angle_indices, angle_bond_indices = sim_state

	pos_indices = np.array_split(np.arange(param['n_bead']), size)[rank]
	frc_indices = (bond_indices[0] + pos_indices[0], bond_indices[1])
	vdw_coeff = np.array_split(param['vdw_matrix'], size)[rank]
	virial_indicies = ut.create_index(np.argwhere(np.array_split(np.tri(param['n_bead']).T, size)[rank]))

	kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
	pressure = 1 / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
	temperature = 2 * kin_energy / n_dof

	if rank == 0:
		tot_pos[0] += np.vstack((pos, cell_dim))
		tot_vel[0] += vel
		tot_frc[0] += frc

		tot_energy[0] = pot_energy + kin_energy
		tot_temp[0] = 2 * kin_energy / n_dof
		tot_press[0] = pressure
		tot_vol[0] = np.prod(cell_dim)

		setup_time_stop = time.time()
		setup_time = setup_time_stop - setup_time_start
		time_hour = int(setup_time / 60**2)
		time_min = int((setup_time / 60) % 60)
		time_sec = int(setup_time) % 60

		print("\n " + " " * 15 + "----Setup Complete----\n")
		print(" {:5d} hr {:2d} min {:2d} sec ({:8.3f} sec)".format(time_hour, time_min, time_sec, setup_time))
		print(" Fibre diameter = {} um\n Simulation cell dimensions = {} um".format(param['l_conv'], cell_dim * param['l_conv']))
		print(" Cell density:     {:>10.4f} bead mass um-3".format(param['n_bead'] * param['mass'] / np.prod(cell_dim * param['l_conv'])))
		print(" Number of Simulation steps = {}".format(param['n_step']))

		sim_time_start = time.time()

		print("\n" + " " * 15 + "----Running Simulation----")

	for step in range(1, param['n_step']):

		sim_state = velocity_verlet_alg_mpi(pos, vel, frc, virial_tensor, param, pos_indices, bond_indices, frc_indices, angle_indices, 
				angle_bond_indices, vdw_coeff, virial_indicies, param['dt']/2, sqrt_dt, cell_dim, calc_energy_forces_mpi, comm, size, rank)

		(pos, vel, frc, cell_dim, pot_energy, virial_tensor) = sim_state

		"""
		"DYNAMIC BONDS - not yet implemented fully"
		if step % 1 == 0: 
			param['bond_matrix'], update = ut.bond_check(param['bond_matrix'], fib_end, r2, param['rc'], param['bond_rb'], param['vdw_sigma'])
			if update:
				bond_beads, dist_index, r_index, fib_end = ut.update_bond_lists(param['bond_matrix'])
				ut.update_param_file(sim_dir + file_names['param_file_name'], 'bond_matrix', param['bond_matrix'])
		#"""

		kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
		pressure = 1 / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))
		temperature = 2 * kin_energy / n_dof
	
		if rank == 0:
			tot_energy[step] += pot_energy + kin_energy
			tot_temp[step] += temperature
			tot_press[step] += pressure
			tot_vol[step] = np.prod(cell_dim)

			if step % param['save_step'] == 0:
				i = int(step / param['save_step'])
				tot_pos[i] += np.vstack((pos, cell_dim))
				tot_vel[i] += vel
				tot_frc[i] += frc

				ut.save_npy(sim_dir + file_names['restart_file_name'], (tot_pos[i], tot_vel[i]))

			if step % param['print_step'] == 0:

				sim_time = (time.time() - sim_time_start) * (param['n_step'] / step - 1) 
				time_hour = int(sim_time / 60**2)
				time_min = int((sim_time / 60) % 60)
				time_sec = int(sim_time) % 60

				print(" " + "-" * 56)
				print(" " + "| Step: {:{dig}d} {}   |".format(step, " " * (44 - dig), dig=dig))
				print(" " + "| Temp: {:>10.4f} kBT    Energy: {:>10.3f} per fibril |".format(tot_temp[step], tot_energy[step] / param['n_fibril']))
				print(" " + "| Pressure: {:>10.4f}    Volume: {:>10.4f}            |".format(tot_press[step], tot_vol[step]))
				print(" " + "|" + " " * 55 + "|")
				print(" " + "| Estimated time remaining: {:5d} hr {:2d} min {:2d} sec      |".format(time_hour, time_min, time_sec))
				print(" " + "-" * 56)

		if temperature >= param['kBT'] * 1E3: 
			if rank == 0: print("velocity exceeded, step ={}".format(step))
			n_step = step
			sys.exit()

	if rank == 0:
		sim_time_stop = time.time()

		sim_time = sim_time_stop - sim_time_start
		time_hour = int(sim_time / 60**2)
		time_min = int((sim_time / 60) % 60)
		time_sec = int(sim_time) % 60

		print("\n " + " " * 15 + "----Simulation Complete----\n")
		print(" {:5d} hr {:2d} min {:2d} sec ({:8.3f} sec)".format(time_hour, time_min, time_sec, sim_time))
		print("\n Averages:")
		print(" Average Temperature: {:>10.4f} kBT".format(np.mean(tot_temp)))
		print(" Average Energy:      {:>10.4f} per fibril".format(np.mean(tot_energy) / param['n_fibril']))
		print(" Average Pressure:    {:>10.4f}".format(np.mean(tot_press)))
		print(" Average Volume:      {:>10.4f}".format(np.mean(tot_vol)))
		print("\n RMS:")
		print(" RMS Temperature: {:>10.4f} kBT".format(np.std(tot_temp)))
		print(" RMS Energy:      {:>10.4f} per fibril".format(np.std(tot_energy / param['n_fibril'])))
		print(" RMS Pressure:    {:>10.4f}".format(np.std(tot_press)))
		print(" RMS Volume:      {:>10.4f}\n".format(np.std(tot_vol)))

		print(" Saving restart file {}".format(file_names['restart_file_name']))
		ut.save_npy(sim_dir + file_names['restart_file_name'], (tot_pos[-1], tot_vel[-1]))

		print(" Saving trajectory file {}".format(file_names['traj_file_name']))
		ut.save_npy(sim_dir + file_names['traj_file_name'], tot_pos)

		print(" Saving output file {}".format(file_names['output_file_name']))
		ut.save_npy(sim_dir + file_names['output_file_name'], (tot_energy, tot_temp, tot_press))


def speed_test(current_dir, comm, input_file_name=False, size=1, rank=0):
	"""
	calc_state(pos, cell_dim, bond_matrix, vdw_matrix, param)
	
	Calculate state of simulation using starting configuration and parameters provided

	Parameters
	----------

	pos:  array_like (float); shape=(n_bead, n_dim)
		Positions of n_bead beads in n_dim

	cell_dim: array_like, dtype=float
		Array with simulation cell dimensions

	bond_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a bond is present between two beads

	vdw_matrix: array_like (int); shape=(n_bead, n_bead)
		Matrix determining whether a non-bonded interaction is present between two beads

	param:  dict
		Dictionary of simulation and analysis parameters
	
	Returns
	-------

	frc: array_like, dtype=float
		Forces acting upon each bead in all collagen fibrils

	verlet_list: array_like, dtype=int
		Matrix determining whether two beads are within rc radial distance

	pot_energy:  float
		Total potential energy of system

	virial_tensor:  array_like, (float); shape=(n_dim, n_dim)
		Virial components of pressure tensor of system

	bond_beads:  array_like, (int); shape=(n_angle, 3)
		Array containing indicies in pos array all 3-bead angular interactions

	dist_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in distance arrays of all bonded interactions

	r_index:  array_like, (int); shape=(n_bond, 2)
		Array containing indicies in r array of all bonded interactions
	
	"""

	import time

	sim_dir = current_dir + '/sim/'

	if rank == 0: 
		setup_time_start = time.time()
		if not os.path.exists(sim_dir): os.mkdir(sim_dir)
		file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name, verbosity=False)
	else:
		file_names = None
		param = None
	file_names = comm.bcast(file_names, root=0)
	param = comm.bcast(param, root=0)

	if param['n_dim'] == 2: import sim_tools_2D as sim
	elif param['n_dim'] == 3: import sim_tools_3D as sim

	n_frames = int(param['n_step'] / param['save_step'])
	dig = len(str(param['n_step']))
	sqrt_dt = np.sqrt(param['dt'])

	pos, vel, cell_dim, param = setup.import_files_mpi(sim_dir, file_names, param, comm, size, rank, verbosity=False)

	n_dof = param['n_dim'] * param['n_bead'] 

	if rank == 0:
		tot_pos = np.zeros((n_frames, param['n_bead'] + 1, param['n_dim']))
		tot_vel = np.zeros((n_frames, param['n_bead'], param['n_dim']))
		tot_frc = np.zeros((n_frames, param['n_bead'], param['n_dim']))

		tot_temp = np.zeros(param['n_step'])
		tot_energy = np.zeros(param['n_step'])
		tot_press = np.zeros(param['n_step'])
		tot_vol = np.zeros(param['n_step'])

	if param['n_dim'] == 2: from sim_tools_2D import calc_energy_forces_mpi
	elif param['n_dim'] == 3: from sim_tools_3D import calc_energy_forces_mpi

	bond_indices, angle_indices, angle_bond_indices = ut.update_bond_lists_mpi(param['bond_matrix'], comm, size, rank)
	
	pos_indices = np.array_split(np.arange(param['n_bead']), size)[rank]
	frc_indices = (bond_indices[0] + pos_indices[0], bond_indices[1])
	vdw_coeff = np.array_split(param['vdw_matrix'], size)[rank]
	virial_indicies = ut.create_index(np.argwhere(np.array_split(np.tri(param['n_bead']).T, size)[rank]))

	calc_times = []
	overhead_times = []
	if ('-ntrial' in sys.argv): n_trial = int(sys.argv[sys.argv.index('-ntrial') + 1])
	else: n_trial = 2000

	for i in range(n_trial):
		start_time = time.time()

		frc, pot_energy, virial_tensor = calc_energy_forces_mpi(pos, cell_dim, pos_indices, bond_indices, frc_indices, 
							angle_indices, angle_bond_indices, vdw_coeff, virial_indicies, param)

		stop_time_1 = time.time()
		calc_times.append(stop_time_1 - start_time)

		pot_energy = np.sum(comm.gather(pot_energy))
		frc = comm.allreduce(frc, op=MPI.SUM)
		virial_tensor = comm.allreduce(virial_tensor, op=MPI.SUM)

		stop_time_2 = time.time()
		overhead_times.append(stop_time_2 - stop_time_1)

	calc_times = np.sum(comm.gather(calc_times, root=0))
	overhead_times = np.sum(comm.gather(overhead_times, root=0))

	if rank == 0:
		calc_times /= (size * n_trial) 
		overhead_times /= (size * n_trial) 
		print(" MPI {} proc energy = {:4.5f}   calc time = {:4.5f} s    mpi overhead time = {:4.5f} s    total time = {:4.5f} s".format(size, pot_energy, calc_times, overhead_times, calc_times + overhead_times))
