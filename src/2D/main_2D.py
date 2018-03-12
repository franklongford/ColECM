"""
COLLAGEN FIBRE SIMULATION 2D

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 09/03/2018
"""

import numpy as np

import sys, os, time

import utilities_2D as ut
import simulation_2D as sim

print(' '+ '_' * 54)
print( "|   ___   ___                        ___   ___         |")
print( "|  /     /   \  |    |       /\     /     |    |\   |  |")
print( "| |     |     | |    |      /  \   |  __  |___ | \  |  |")
print( "| |     |     | |    |     /----\  |    | |    |  \ |  |")
print( "|  \___  \___/  |___ |___ /      \  \___/ |___ |   \|  |")
print( '|'+ '_' * 54 + '|' + '  v1.0.0.dev1')
print( "\n          ECM Collagen Fibre Simulation\n")

current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))


if ('-n' in sys.argv): n_steps = int(sys.argv[sys.argv.index('-n') + 1])
else: n_steps = 10000

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

n_dim = 2
traj_steps = 100
dt = 0.002

pos, cell_dim, bond_matrix, params = sim.import_files(n_dim, param_file_name, pos_file_name)
if len(params) == 7: mass, vdw_param, bond_param, angle_param, rc, kBT, Langevin = params
else: mass, vdw_param, bond_param, angle_param, rc, kBT, Langevin, thermo_gamma, thermo_sigma = params

n_bead = pos.shape[0]

vel, frc, verlet_list, bond_beads, dxdy_index, r_index = sim.setup(pos, cell_dim, bond_matrix, mass, vdw_param, bond_param, angle_param, rc, kBT)

tot_pos = np.zeros((int(n_steps/traj_steps), n_bead, n_dim))
tot_vel = np.zeros((int(n_steps/traj_steps), n_bead, n_dim))
tot_frc = np.zeros((int(n_steps/traj_steps), n_bead, n_dim))

#energy_array = np.zeros(n_steps)

init_time_stop = time.time()

print("\nSetup complete: {:5.3f} s \nSimulation cell dimensions = {}".format(init_time_stop - init_time_start, cell_dim))

sim_time_start = time.time()

dx, dy = sim.get_dx_dy(pos, cell_dim)
r2 = dx**2 + dy**2
verlet_list = sim.check_cutoff(r2, rc**2)

print("\nRunning Simulation")

for step in range(n_steps):
	sys.stdout.write("STEP {}\r".format(step))
	sys.stdout.flush()

	thermo_xi = np.random.normal(0, 1, (n_bead, n_dim))
	thermo_theta = np.random.normal(0, 1, (n_bead, n_dim))

	pos, vel, frc, verlet_list, energy = sim.velocity_verlet_alg(pos, vel, frc, mass, bond_matrix, verlet_list,
						bond_beads, dxdy_index, r_index, dt, cell_dim, vdw_param, bond_param,
						angle_param, rc, kBT, Langevin, thermo_gamma, thermo_sigma, thermo_xi, thermo_theta)

	#energy_array[step] += energy

	if step % traj_steps == 0:
		i = int(step / traj_steps)
		tot_pos[i] += pos
		tot_vel[i] += vel
		tot_frc[i] += frc

	if np.sum(np.abs(vel)) >= kBT * 1E5: 
		print("velocity exceeded, step ={}".format(step))
		dx, dy = sim.get_dx_dy(pos, cell_dim)
		r2 = dx**2 + dy**2
		n_steps = step
		break 
	

sim_time_stop = time.time()

sim_time = sim_time_stop - sim_time_start

print("\nSimulation complete: {:10d} min {:2d} s ({:5.3f} s)".format(int(sim_time / 60), int(sim_time) % 60, sim_time))

print("Saving restart file {}".format(restart_file_name))
ut.save_npy(restart_file_name, tot_pos[-1])

print("Saving trajectory file {}".format(traj_file_name))
ut.save_npy(traj_file_name, tot_pos)

#CMA = ut.cum_mov_average(energy_array[:n_steps]) / n_bead
#plt.plot(CMA)
#plt.show()


