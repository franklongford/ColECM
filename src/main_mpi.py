"""
ColECM: Collagen ExtraCellular Matrix Simulation
MAIN MPI ROUTINE 

Created by: Frank Longford
Created on: 31/12/2018

Last Modified: 31/12/2018
"""

import sys, os
import utilities as ut
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

modules = []

if ('simulation' in sys.argv): modules.append('simulation') 
if ('analysis' in sys.argv): modules.append('analysis')
if ('editor' in sys.argv): modules.append('editor')
if ('speed' in sys.argv): modules.append('speed')
else:
	if rank == 0: 
		ut.logo()
		print(" Running on {} processors\n".format(size))

if len(modules) == 0: 
	if rank == 0: modules = (input(' Please enter desired modules to run (SIMULATION and/or ANALYSIS or EDITOR): ').lower()).split()
	else: modules = False
	modules = comm.bcast(modules, root=0)

if ('-input' in sys.argv): input_file_name = current_dir + '/' + sys.argv[sys.argv.index('-input') + 1]
else: input_file_name = False

if ('simulation' in modules):
	from simulation_mpi import simulation
	simulation(current_dir, comm, input_file_name, size, rank)
if ('analysis' in modules):
	from analysis_mpi import analysis
	analysis(current_dir, comm, input_file_name, size, rank)
if ('editor' in modules):
	from editor_mpi import editor
	editor(current_dir, comm, input_file_name, size, rank)
if ('speed' in modules):
	from simulation_mpi import speed_test
	speed_test(current_dir, comm, input_file_name, size, rank)
