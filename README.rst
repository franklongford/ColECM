==================================================
ColECM - Collagen Extracellular Matrix Simulation
==================================================

By Frank Longford (2018)
------------------------

Simulation code designed to reproduce second harmonic generation (SHG) images of the collagen fibril network in the extra cellular matrix (ECM). Details on the model used can be found in documentation.pdf.

Installation:
-------------

run ``python install.py``

Note: system must have working versions of Python and pip for python >= 3.0. Installation will check for common executables (``python, python3``) - please edit ``PYTHON`` and ``PIP`` macros on Makefile if local installations are named differently.


Instructions:
-------------

Main program and analysis can be run via the following commands:

1) ``./ColECM [flags]``

	Runs the main simulation routine for collagen fibrils in the ECM. Each fibril is approximated to be 1 um long.
	Flags are optional and can be entered in the command line or selected later during the setup process. 
	All input and output files will be created in the user's current working directory.

		-pos		Name of input position file 
				(will create a new file ending in '_pos' if does not exist)
		-param		Name of parameter file 
				(will create a new file ending in '_param' if does not exist)
		-out		Name of output files (restart and trajectory)
		-ndim		Number of dimensions (2 or 3)
		-nstep		Number of timesteps in simulation (default=10000)
		-mass		Mass of each of collagen bead in red. units
		-vdw_sigma	Van de Waals radius of collagen beads in red. units
		-vdw_epsilon	Van de Waals energy in red. units
		-bond_k		Harmonic bond energy in red. units
		-angle_k	Sigmoidal angle energy in red. units
		-Langevin	Whether to use a Langevin thermostat (NVT) or not (NVE)
		-kBT		Temperature constant in red. units
		-thermo_gamma	Value of Langevin collision rate gamma in red. units (0-1)
		-nfibrilx	Number of collagen fibrils accross x axis
		-nfibrily	Number of collagen fibrils accross y axis
		-nfibrilz	Number of collagen fibrils accross z axis (3D only)
		-lfibril	Length of collagen fibrils in beads

2) ``python3 src/analysis.py [flags]``

	Anisotropy analysis
	Flags are optional and can be entered in the command line or selected later during the analysis process.

		-param	Name of simulation parameter file
		-traj	Name of trajectory file to analyse
		-out	Name of simulation output file to analyse
		-gif	Name of gif file to be created
		-res	Image resolution parameter (1-10)
		-sharp	Image sharpness parameter (1-10)
		-skip	Number of sampled frames between each png
