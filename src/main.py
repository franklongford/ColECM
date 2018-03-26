"""
COLLAGEN FIBRE SIMULATION

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 09/03/2018
"""

import numpy as np
import sys, os, time

import utilities as ut
import setup

print(' '+ '_' * 54)
print( "|   ___   ___                        ___   ___         |")
print( "|  /     /   \  |    |       /\     /     |    |\   |  |")
print( "| |     |     | |    |      /  \   |  __  |___ | \  |  |")
print( "| |     |     | |    |     /----\  |    | |    |  \ |  |")
print( "|  \___  \___/  |___ |___ /      \  \___/ |___ |   \|  |")
print( '|'+ '_' * 54 + '|' + '  v1.0.0.dev1')
print( "\n          ECM Collagen Fibre Simulation\n")


if ('-n_dim' in sys.argv): n_dim = int(sys.argv[sys.argv.index('-n_dim') + 1])
else: n_dim = int(input("Number of dimensions: "))

assert n_dim in [2, 3]

if n_dim == 2: import simulation_2D as sim
elif n_dim == 3: import simulation_3D as sim

current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

if ('-n_step' in sys.argv): n_steps = int(sys.argv[sys.argv.index('-n_step') + 1])
else: n_step = 10000

if ('-param' in sys.argv): param_file_name = current_dir + '/' + sys.argv[sys.argv.index('-param') + 1]
else: param_file_name = current_dir + '/' + input("Enter param_file name: ")

if ('-pos' in sys.argv): pos_file_name = current_dir + '/' + sys.argv[sys.argv.index('-pos') + 1]
else: pos_file_name = current_dir + '/' + input("Enter pos_file name: ")

if ('-out' in sys.argv): output_file_name = current_dir + '/' + sys.argv[sys.argv.index('-out') + 1]
else: output_file_name = current_dir + '/' + input("Enter output_file name: ")

param_file_name = param_file_name + '_param'
traj_file_name = output_file_name + '_traj'
restart_file_name = output_file_name + '_rst'

print("\nEntering Setup\n")
init_time_start = time.time()

traj_steps = 100
n_frames = int(n_step / traj_steps)
dt = 0.004

pos, cell_dim, l_conv, bond_matrix, vdw_matrix, params = setup.import_files(n_dim, param_file_name, pos_file_name)

if len(params) == 7: mass, vdw_param, bond_param, angle_param, rc, kBT, Langevin = params
else: mass, vdw_param, bond_param, angle_param, rc, kBT, Langevin, thermo_gamma, thermo_sigma = params

n_bead = pos.shape[0]

vel, frc, verlet_list, bond_beads, dxdy_index, r_index = setup.initial_state(n_dim, pos, cell_dim, bond_matrix, vdw_matrix,
																 vdw_param, bond_param, angle_param, rc, kBT)

tot_pos = np.zeros((n_frames, n_bead, n_dim))
tot_vel = np.zeros((n_frames, n_bead, n_dim))
tot_frc = np.zeros((n_frames, n_bead, n_dim))
tot_temp = np.zeros((n_frames))

energy_array = np.zeros(n_step)

init_time_stop = time.time()

print("\nSetup complete: {:5.3f} s \nBead radius = {} um, Simulation cell dimensions = {} um".format(
	init_time_stop - init_time_start, l_conv, cell_dim * l_conv))

sim_time_start = time.time()

distances = ut.get_distances(pos, cell_dim)
r2 = np.sum(distances**2, axis=0)
verlet_list = ut.check_cutoff(r2, rc**2)

print("\nRunning Simulation")

for step in range(n_step):
	sys.stdout.write("STEP {}\r".format(step))
	sys.stdout.flush()

	thermo_xi = np.random.normal(0, 1, (n_bead, n_dim))
	thermo_theta = np.random.normal(0, 1, (n_bead, n_dim))

	pos, vel, frc, verlet_list, energy = sim.velocity_verlet_alg(n_dim, pos, vel, frc, mass, bond_matrix, vdw_matrix, verlet_list,
						bond_beads, dxdy_index, r_index, dt, cell_dim, vdw_param, bond_param,
						angle_param, rc, kBT, Langevin, thermo_gamma, thermo_sigma, thermo_xi, thermo_theta)

	energy_array[step] += energy

	if step % traj_steps == 0:
		i = int(step / traj_steps)
		tot_pos[i] += pos
		tot_vel[i] += vel
		tot_frc[i] += frc
		tot_temp[i] += ut.kin_energy(vel, mass, n_dim) * 2

	if np.sum(np.abs(vel)) >= kBT * 1E5: 
		print("velocity exceeded, step ={}".format(step))
		n_step = step
		sys.exit() 

print("Min Velocity: {:4.5f}".format(np.min(abs(tot_vel))))
print("Max Velocity:  {:4.5f}".format(np.max(abs(tot_vel))))
print("Average Velocity: {:4.5f}".format(np.mean(abs(tot_vel))))
print("Average Temperature: {:4.5f}".format(np.mean(tot_temp)))
print("Average Energy: {:4.5f} per bead".format(np.mean(energy_array) / n_bead))	

sim_time_stop = time.time()

sim_time = sim_time_stop - sim_time_start
time_hour = int(sim_time / 60**2)
time_min = int((sim_time / 60) % 60)
time_sec = int(sim_time) % 60

print("\nSimulation complete: {:5d} hr {:2d} min {:2d} sec ({:8.3f} sec)".format(time_hour, time_min, time_sec, sim_time))

print("Saving restart file {}".format(restart_file_name))
ut.save_npy(restart_file_name, tot_pos[-1])

print("Saving trajectory file {}".format(traj_file_name))
ut.save_npy(traj_file_name, tot_pos)
