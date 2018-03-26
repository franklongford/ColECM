===============================================
Collagen - Pythonic collagen fibril simulation
===============================================

By Frank Longford (2018)
------------------------

Simulation code designed to reproduce SHG images of the colagen fibril network in the extra cellular matrix (ECM).

Installation:
-------------

``python3 setup.py install``


Instructions:
-------------

Main program and analysis can be run via the following commands:

1) ``python3 src/main.py [flags]``

	Flags are optional and can be entered in the command line or selected later during the setup process. 
	All input and output files will be created in the user's current working directory.

		-pos		Name of input position file (will create a new one if does not exist)
		-param		Name of parameter file (will create a new one if does not exist)
		-out		Name of output files (restart and trajectory)
		- n_dim		Number of dimensions (must be 2 or 3)
		- n_step		Number of timesteps in simulation (default=10000)
		-vdw_sigma	Van de Waals radius of collagen beads in red. units
		-vdw_epsilon	Van de Waals minimul energy in red. units
		-bond_k		Harmonic bond energy in red. units
		-angle_k	Sigmoidal angle energy in red. units
		-kBT		Temperature constant in red. units
		-Langevin	Whether to use Langevin dynamics (Y/N)
		-thermo_gamma	Value of Langevin collision rate gamma in red. units
		- n_fibril	Number of collagen fibrils accross one axis
		- l_fibril	Length of collagen fibrils in beads
		- n_layer	Number of repeating unit cells along z axis (3D only)

2) ``python3 src/analysis.py [flags]``

	Flags are optional and can be entered in the command line or selected later during the analysis process.

		-traj	Name of trajectory file to analyse
		-param	Name of parameter file
		-gif	Name of gif file to be created
		-res	Image resolution parameter (1-10)
		-sharp	Image sharpness parameter (1-10)
