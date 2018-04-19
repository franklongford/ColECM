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

def simulation(current_dir, dir_path):	

	print("\nEntering Setup\n")
	init_time_start = time.time()

	sim_dir = current_dir + '/sim/'
	if not os.path.exists(sim_dir): os.mkdir(sim_dir)

	file_names, param = setup.read_shell_input(current_dir, sim_dir)

	if param['n_dim'] == 2: 
		import sim_tools_2D as sim
		dt = 0.004
	elif param['n_dim'] == 3: 
		import sim_tools_3D as sim
		dt = 0.003

	n_frames = int(param['n_step'] / param['save_step'])
	dig = len(str(param['n_step']))
	sqrt_dt = np.sqrt(dt)

	param_file_name, pos_file_name, traj_file_name, restart_file_name, output_file_name, _ = file_names
	pos, vel, cell_dim, param = setup.import_files(sim_dir, file_names, param)

	vdw_param = (param['vdw_sigma'], param['vdw_epsilon'])
	bond_param = (param['bond_r0'], param['bond_k'])
	angle_param = (param['angle_theta0'], param['angle_k'])

	n_bead = pos.shape[0]
	frc, verlet_list, bond_beads, dxdy_index, r_index = setup.initial_state(pos, cell_dim, param['bond_matrix'], 
		param['vdw_matrix'], vdw_param, bond_param, angle_param, param['rc'], param['kBT'])

	tot_pos = np.zeros((n_frames, n_bead + 1, param['n_dim']))
	tot_vel = np.zeros((n_frames, n_bead, param['n_dim']))
	tot_frc = np.zeros((n_frames, n_bead, param['n_dim']))

	tot_temp = np.zeros(param['n_step'])
	tot_energy = np.zeros(param['n_step'])

	init_time_stop = time.time()

	print("\nSetup complete: {:5.3f} s".format(init_time_stop - init_time_start))
	print("Number of beads = {}".format(n_bead))
	print("Bead radius = {} um\nSimulation cell dimensions = {} um".format(param['l_conv'], cell_dim * param['l_conv']))
	print("Number of Simulation steps = {}".format(param['n_step']))

	sim_time_start = time.time()

	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)
	verlet_list = ut.check_cutoff(r2, param['rc']**2)

	print("\n" + " " * 15 + "----Running Simulation----")

	for step in range(param['n_step']):

		pos, vel, frc, verlet_list, energy = sim.velocity_verlet_alg(param['n_dim'], pos, vel, frc, param['mass'], 
			param['bond_matrix'], param['vdw_matrix'], verlet_list, bond_beads, dxdy_index, r_index, dt, sqrt_dt, cell_dim, 
			vdw_param, bond_param, angle_param, param['rc'], param['kBT'], param['gamma'], param['sigma'])

		tot_energy[step] += energy
		tot_temp[step] += ut.kin_energy(vel, param['mass'], param['n_dim']) * 2

		if step % param['save_step'] == 0:
			i = int(step / param['save_step'])
			tot_pos[i] += np.vstack((pos, cell_dim))
			tot_vel[i] += vel
			tot_frc[i] += frc

			ut.save_npy(sim_dir + restart_file_name, (tot_pos[i], tot_vel[i]))

			sys.stdout.write(" " * 65 + "\r")
			sys.stdout.flush()
			print(" " + "-" * 55)
			print(" " + "| Step: {:{dig}d} {} |".format(step, " " * (44 - dig), dig=dig))
			print(" " + "| Temp: {:>10.4f} / kBT    Energy: {:>10.4f} / bead |".format(tot_temp[step] / param['kBT'], tot_energy[step] / n_bead))
			print(" " + "-" * 55)

		if np.max(np.abs(vel)) >= param['kBT'] * 1E5: 
			print("velocity exceeded, step ={}".format(step))
			n_step = step
			sys.exit()

		if step > 0: 
			sim_time = (time.time() - sim_time_start) * (param['n_step'] / step - 1) 
			time_hour = int(sim_time / 60**2)
			time_min = int((sim_time / 60) % 60)
			time_sec = int(sim_time) % 60

			sys.stdout.write(" Estimated time remaining: {:5d} hr {:2d} min {:2d} sec ({:8.3f} sec)\r".format(time_hour, time_min, time_sec, sim_time))
			sys.stdout.flush()

	sim_time_stop = time.time()

	sim_time = sim_time_stop - sim_time_start
	time_hour = int(sim_time / 60**2)
	time_min = int((sim_time / 60) % 60)
	time_sec = int(sim_time) % 60

	print("\n----Simulation Complete----\n")
	print("{:5d} hr {:2d} min {:2d} sec ({:8.3f} sec)".format(time_hour, time_min, time_sec, sim_time))
	print("\nAverages:")
	print("Average Temperature: {:>10.4f} / kBT".format(np.mean(tot_temp) / param['kBT']))
	print("Average Energy:      {:>10.4f} / bead".format(np.mean(tot_energy) / n_bead))
	print("\nRMS:")
	print("RMS Temperature: {:>10.4f} / kBT".format(np.std(tot_temp / param['kBT'])))
	print("RMS Energy:      {:>10.4f} / bead\n".format(np.std(tot_energy / n_bead)))	

	print("Saving restart file {}".format(restart_file_name))
	ut.save_npy(sim_dir + restart_file_name, (tot_pos[-1], tot_vel[-1]))

	print("Saving trajectory file {}".format(traj_file_name))
	ut.save_npy(sim_dir + traj_file_name, tot_pos)

	print("Saving output file {}".format(output_file_name))
	ut.save_npy(sim_dir + output_file_name, (tot_energy, tot_temp))
