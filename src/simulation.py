"""
ColECM: Collagen ExtraCellular Matrix Simulation
SIMULATION ROUTINE 

Created by: Frank Longford
Created on: 13/04/2018

Last Modified: 12/04/2018
"""

import numpy as np
import sys, os, time

import utilities as ut
import setup

def simulation(current_dir, dir_path):

	if ('-ndim' in sys.argv): n_dim = int(sys.argv[sys.argv.index('-ndim') + 1])
	else: n_dim = int(input("Number of dimensions: "))

	assert n_dim in [2, 3]

	if n_dim == 2: import sim_tools_2D as sim
	elif n_dim == 3: import sim_tools_3D as sim

	if ('-nstep' in sys.argv): n_step = int(sys.argv[sys.argv.index('-nstep') + 1])
	else: n_step = 10000

	if ('-param' in sys.argv): param_file_name = current_dir + '/' + sys.argv[sys.argv.index('-param') + 1]
	else: param_file_name = current_dir + '/' + input("Enter param_file name: ")

	if ('-pos' in sys.argv): pos_file_name = current_dir + '/' + sys.argv[sys.argv.index('-pos') + 1]
	else: pos_file_name = current_dir + '/' + input("Enter pos_file name: ")

	if ('-traj' in sys.argv): traj_file_name = current_dir + '/' + sys.argv[sys.argv.index('-traj') + 1]
	else: traj_file_name = current_dir + '/' + input("Enter traj_file name: ")

	if ('-rst' in sys.argv): restart_file_name = current_dir + '/' + sys.argv[sys.argv.index('-rst') + 1]
	else: restart_file_name = current_dir + '/' + input("Enter restart_file name: ")

	if ('-out' in sys.argv): output_file_name = current_dir + '/' + sys.argv[sys.argv.index('-out') + 1]
	else: output_file_name = current_dir + '/' + input("Enter output_file name: ")

	param_file_name = ut.check_file_name(param_file_name, 'param', 'pkl') + '_param'
	pos_file_name = ut.check_file_name(pos_file_name, extension='npy')
	traj_file_name = ut.check_file_name(traj_file_name, 'traj', 'npy') + '_traj'
	restart_file_name = ut.check_file_name(restart_file_name, 'rst', 'npy') + '_rst'
	output_file_name = ut.check_file_name(output_file_name, 'out', 'npy') + '_out'

	print("\nEntering Setup\n")
	init_time_start = time.time()

	traj_steps = 100
	n_frames = int(n_step / traj_steps)
	dig = len(str(n_step))
	if n_dim == 2: dt = 0.004
	elif n_dim == 3: dt = 0.003

	sqrt_dt = np.sqrt(dt)

	pos, vel, cell_dim, l_conv, bond_matrix, vdw_matrix, params = setup.import_files(n_dim, param_file_name, pos_file_name, restart_file_name)

	mass, vdw_param, bond_param, angle_param, rc, Langevin, kBT, thermo_gamma = params

	n_bead = pos.shape[0]

	frc, verlet_list, bond_beads, dxdy_index, r_index = setup.initial_state(n_dim, pos, cell_dim, bond_matrix, vdw_matrix,
										vdw_param, bond_param, angle_param, rc, kBT)

	tot_pos = np.zeros((n_frames, n_bead + 1, n_dim))
	tot_vel = np.zeros((n_frames, n_bead, n_dim))
	tot_frc = np.zeros((n_frames, n_bead, n_dim))

	tot_temp = np.zeros(n_step)
	tot_energy = np.zeros(n_step)

	init_time_stop = time.time()

	print("\nSetup complete: {:5.3f} s".format(init_time_stop - init_time_start))
	print("Number of beads = {}".format(n_bead))
	print("Bead radius = {} um\nSimulation cell dimensions = {} um".format(l_conv, cell_dim * l_conv))

	sim_time_start = time.time()

	distances = ut.get_distances(pos, cell_dim)
	r2 = np.sum(distances**2, axis=0)
	verlet_list = ut.check_cutoff(r2, rc**2)

	if Langevin: thermo_sigma = np.sqrt(thermo_gamma * (2 - thermo_gamma) * (kBT / mass))
	else: thermo_sigma = 0

	print("\n----Running Simulation----")

	for step in range(n_step):

		pos, vel, frc, verlet_list, energy = sim.velocity_verlet_alg(n_dim, pos, vel, frc, mass, bond_matrix, vdw_matrix, verlet_list,
										bond_beads, dxdy_index, r_index, dt, sqrt_dt, cell_dim, vdw_param, bond_param, angle_param, 
										rc, kBT, thermo_gamma, thermo_sigma)

		tot_energy[step] += energy
		tot_temp[step] += ut.kin_energy(vel, mass, n_dim) * 2

		if step % traj_steps == 0:
			i = int(step / traj_steps)
			tot_pos[i] += np.vstack((pos, cell_dim))
			tot_vel[i] += vel
			tot_frc[i] += frc

			ut.save_npy(restart_file_name, (tot_pos[step], tot_vel[step]))

			print("-" * 55)
			print("| Step: {:{dig}d} {} |".format(step, " " * (44 - dig), dig=dig))
			print("| Temp: {:>10.4f} / kBT    Energy: {:>10.4f} / bead |".format(tot_temp[step] / kBT, tot_energy[step] / n_bead))
			print("-" * 55)

		if np.max(np.abs(vel)) >= kBT * 1E5: 
			print("velocity exceeded, step ={}".format(step))
			n_step = step
			sys.exit() 

	sim_time_stop = time.time()

	sim_time = sim_time_stop - sim_time_start
	time_hour = int(sim_time / 60**2)
	time_min = int((sim_time / 60) % 60)
	time_sec = int(sim_time) % 60

	print("\n----Simulation Complete----\n")
	print("{:5d} hr {:2d} min {:2d} sec ({:8.3f} sec)".format(time_hour, time_min, time_sec, sim_time))
	print("\nAverages:")
	print("Average Velocity:    {:>10.4f}".format(np.mean(abs(tot_vel))))
	print("Average Temperature: {:>10.4f} / kBT".format(np.mean(tot_temp) / kBT))
	print("Average Energy:      {:>10.4f} / bead".format(np.mean(tot_energy) / n_bead))
	print("\nRMS:")
	print("RMS Velocity:    {:>10.4f}".format(np.std(tot_vel)))
	print("RMS Temperature: {:>10.4f} / kBT".format(np.std(tot_temp / kBT)))
	print("RMS Energy:      {:>10.4f} / bead\n".format(np.std(tot_energy / n_bead)))	

	print("Saving restart file {}".format(restart_file_name))
	ut.save_npy(restart_file_name, (tot_pos[-1], tot_vel[-1]))

	print("Saving trajectory file {}".format(traj_file_name))
	ut.save_npy(traj_file_name, tot_pos)

	print("Saving output file {}".format(output_file_name))
	ut.save_npy(output_file_name, (tot_energy, tot_temp))
