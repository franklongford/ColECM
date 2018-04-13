"""
ColECM: Collagen ExtraCellular Matrix Simulation
MAIN ROUTINE 

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 12/04/2018
"""

import sys, os
import utilities as ut

ut.logo()
current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

modules = []

if ('simulation' in sys.argv): modules.append('simulation') 
elif ('analysis' in sys.argv): modules.append('analysis')
else: modules = (input('Please enter desired modules to run (SIMULATION and/or ANALYSIS): ').lower()).split()

if ('simulation' in modules):
	from simulation import simulation
	simulation(current_dir, dir_path)
if ('analysis' in modules):
	from analysis import analysis
	analysis(current_dir, dir_path)
