"""
COLLAGEN FIBRE SIMULATION 2D

Created by: Frank Longford
Created on: 01/11/15

Last Modified: 06/03/2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys, os, time, pathlib

import utilities_2D as ut


current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv) < 2: param_file_name = current_dir + '/' + input("Enter param_file name: ")
else: param_file_name = current_dir + '/' + sys.argv[1]
if len(sys.argv) < 3: pos_file_name = current_dir + '/' + input("Enter pos_file name: ")
else: pos_file_name = current_dir + '/' + sys.argv[2]
if len(sys.argv) < 4: output_file_name = current_dir + '/' + input("Enter output_file name: ")
else: output_file_name = current_dir + '/' + sys.argv[3]

traj_file_name = output_file_name + '_traj'
restart_file_name = output_file_name + '_rst'

print(param_file_name)

print("Entering Setup")
init_time_start = time.time()

n_dim = 2
n_steps = 10000
traj_steps = 100
dt = 0.002

if not os.path.exists('{}.pkl'.format(param_file_name)): 

	ut.make_param_file(param_file_name)

	mass = float(input("Enter bead mass: "))
	param_file = ut.update_param_file(param_file_name, 'mass', mass)

	vdw_sigma = float(input("Enter vdw sigma radius: "))
	vdw_epsilon = float(input("Enter vdw epsilon energy: "))
	vdw_param = [vdw_sigma, vdw_epsilon]
	param_file = ut.update_param_file(param_file_name, 'vdw_param', vdw_param)

	bond_r0 = 2.**(1./6.) * vdw_sigma
	bond_k = float(input("Enter bond k energy: "))
	bond_param = [bond_r0, bond_k]
	param_file = ut.update_param_file(param_file_name, 'bond_param', bond_param)

	angle_theta0 = np.pi
	angle_k = float(input("Enter angle k energy: "))
	angle_param = [angle_theta0, angle_k]
	param_file = ut.update_param_file(param_file_name, 'angle_param', angle_param)

	rc = 4 * vdw_sigma
	param_file = ut.update_param_file(param_file_name, 'rc', rc)

	Langevin = bool(input("Langevin thermostat? (Y/N) ").upper() == 'Y')
	param_file = ut.update_param_file(param_file_name, 'Langevin', Langevin)

	if Langevin: 
		kBT = float(input("Enter kBT constant: "))
		param_file = ut.update_param_file(param_file_name, 'kBT', kBT)

		thermo_gamma = float(input("Enter Langevin gamma constant: "))
		param_file = ut.update_param_file(param_file_name, 'thermo_gamma', thermo_gamma)

		thermo_sigma =  np.sqrt(2 * kBT * thermo_gamma / mass)
		param_file = ut.update_param_file(param_file_name, 'thermo_sigma', thermo_sigma)

else:		
	param_file = ut.read_param_file(param_file_name)

	mass = param_file['mass']
	vdw_param = param_file['vdw_param']
	bond_param = param_file['bond_param']
	angle_param = param_file['angle_param']
	rc = param_file['rc']
	Langevin = param_file['Langevin']

	if Langevin: 
		kBT = param_file['kBT']
		thermo_gamma = param_file['thermo_gamma']
		thermo_sigma = param_file['thermo_sigma']


if not os.path.exists(pos_file_name + '.npy'):
	n_fibre = int(input("Enter square root of number of fibrils: "))
	n_fibre *= n_fibre
	l_fibre = int(input("Enter length of fibril (no. of beads): "))

	pos, cell_dim, bond_matrix = ut.create_pos_array(n_dim, n_fibre, l_fibre, vdw_param, bond_param, angle_param, rc)
	print("Saving input pos file {}.npy".format(pos_file_name))
	ut.save_npy(pos_file_name, pos)

	param_file = ut.update_param_file(param_file_name, 'cell_dim', cell_dim)
	param_file = ut.update_param_file(param_file_name, 'bond_matrix', bond_matrix)
	
else:
	print("Loading input pos file {}.npy".format(pos_file_name))
	pos = ut.load_npy(pos_file_name)
	cell_dim = param_file['cell_dim']
	bond_matrix = param_file['bond_matrix']

n_bead = pos.shape[0]

vel, frc, verlet_list, bond_beads, dxdy_index, r_index = ut.setup(pos, cell_dim, bond_matrix, mass, vdw_param, bond_param, angle_param, rc, kBT)

tot_pos = np.zeros((int(n_steps/traj_steps), n_bead, n_dim))
tot_vel = np.zeros((int(n_steps/traj_steps), n_bead, n_dim))
tot_frc = np.zeros((int(n_steps/traj_steps), n_bead, n_dim))

energy_array = np.zeros(n_steps)

init_time_stop = time.time()

print("Setup complete: {:5.3f} s \nSimulation cell dimensions = {}".format(init_time_stop - init_time_start, cell_dim))

sim_time_start = time.time()

dx, dy = ut.get_dx_dy(pos, n_bead, cell_dim)
r2 = dx**2 + dy**2
verlet_list = ut.check_cutoff(r2, rc**2)

print("Running Simulation")

for step in range(n_steps):
	sys.stdout.write("STEP {}\r".format(step))
	sys.stdout.flush()

	thermo_xi = np.random.normal(0, 1, (n_bead, n_dim))
	thermo_theta = np.random.normal(0, 1, (n_bead, n_dim))

	pos, vel, frc, verlet_list, energy = ut.velocity_verlet_alg(pos, vel, frc, mass, bond_matrix, verlet_list,
						bond_beads, dxdy_index, r_index, dt, cell_dim, vdw_param, bond_param,
						angle_param, rc, kBT, thermo_gamma, thermo_sigma, thermo_xi, thermo_theta, Langevin)

	energy_array[step] += energy

	if step % traj_steps == 0:
		i = int(step / traj_steps)
		tot_pos[i] += pos
		tot_vel[i] += vel
		tot_frc[i] += frc

	if np.sum(np.abs(vel)) >= kBT * 1E5: 
		print("velocity exceeded, step ={}".format(step))
		dx, dy = ut.get_dx_dy(pos, n_bead, cell_dim)
		r2 = dx**2 + dy**2
		n_steps = step
		break 
	

sim_time_stop = time.time()

sim_time = sim_time_stop - sim_time_start

print("\nSimulation complete: {:10d} min {:3d} s ({:5.3f} s)".format(int(sim_time / 60), int(sim_time) % 60, sim_time))

print("Saving restart file {}".format(restart_file_name))
ut.save_npy(restart_file_name, tot_pos[-1])

print("Saving trajectory file {}".format(traj_file_name))
ut.save_npy(traj_file_name, tot_pos)

CMA = ut.cum_mov_average(energy_array[:n_steps]) / n_bead
plt.plot(CMA)
plt.show()


