"""
ColECM: Collagen ExtraCellular Matrix Simulation
SIMULATION ROUTINE 

Created by: Frank Longford
Created on: 13/04/2018

Last Modified: 19/04/2018
"""

import numpy as np
import sys, os, time

import utilities as ut
import setup

def simulation(current_dir, input_file_name=False):	

	print("\n Entering Setup\n")
	init_time_start = time.time()

	sim_dir = current_dir + '/sim/'
	if not os.path.exists(sim_dir): os.mkdir(sim_dir)

	file_names, param = setup.read_shell_input(current_dir, sim_dir, input_file_name)

	if param['n_dim'] == 2: import sim_tools_2D as sim
	elif param['n_dim'] == 3: import sim_tools_3D as sim

	n_frames = int(param['n_step'] / param['save_step'])
	dig = len(str(param['n_step']))
	sqrt_dt = np.sqrt(param['dt'])

	pos, vel, cell_dim, param = setup.import_files(sim_dir, file_names, param)

	n_dof = param['n_dim'] * param['n_bead'] 

	tot_pos = np.zeros((n_frames, param['n_bead'] + 1, param['n_dim']))
	tot_vel = np.zeros((n_frames, param['n_bead'], param['n_dim']))
	tot_frc = np.zeros((n_frames, param['n_bead'], param['n_dim']))

	tot_temp = np.zeros(param['n_step'])
	tot_energy = np.zeros(param['n_step'])
	tot_press = np.zeros(param['n_step'])
	tot_vol = np.zeros(param['n_step'])

	sim_state = setup.calc_state(pos, vel, cell_dim, param['bond_matrix'], param['vdw_matrix'], param)
	frc, verlet_list_rc, pot_energy, virial_tensor, bond_beads, dist_index, r_index, fib_end = sim_state

	tot_pos[0] += np.vstack((pos, cell_dim))
	tot_vel[0] += vel
	tot_frc[0] += frc

	kin_energy = ut.kin_energy(vel, param['mass'], param['n_dim'])
	pressure = 1 / (np.prod(cell_dim) * param['n_dim']) * (kin_energy - 0.5 * np.sum(np.diag(virial_tensor)))

	tot_energy[0] = pot_energy + kin_energy
	tot_temp[0] = 2 * kin_energy / n_dof
	tot_press[0] = pressure
	tot_vol[0] = np.prod(cell_dim)

	init_time_stop = time.time()

	print("\n Setup complete: {:5.3f} s".format(init_time_stop - init_time_start))
	print(" Fibre diameter = {} um\n Simulation cell dimensions = {} um".format(param['l_conv'], cell_dim * param['l_conv']))
	print(" Cell density:     {:>10.4f} bead mass um-3".format(param['n_bead'] * param['mass'] / np.prod(cell_dim * param['l_conv'])))
	print(" Number of Simulation steps = {}".format(param['n_step']))

	sim_time_start = time.time()

	print("\n" + " " * 15 + "----Running Simulation----")

	for step in range(1, param['n_step']):

		sim_state = sim.velocity_verlet_alg(pos, vel, frc, virial_tensor, param, 
			param['bond_matrix'], param['vdw_matrix'], verlet_list_rc, bond_beads, dist_index, 
			r_index, param['dt'], sqrt_dt, cell_dim)

		(pos, vel, frc, cell_dim, pot_energy, virial_tensor, r2) = sim_state

		verlet_list_rc = ut.check_cutoff(r2, param['rc']**2)

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
	
		tot_energy[step] += pot_energy + kin_energy
		tot_temp[step] += 2 * kin_energy / n_dof
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

		if tot_temp[step] >= param['kBT'] * 1E3: 
			print("velocity exceeded, step ={}".format(step))
			n_step = step
			sys.exit()

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
