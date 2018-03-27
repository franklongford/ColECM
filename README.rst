===============================================
Collagen - Pythonic collagen fibril simulation
===============================================

By Frank Longford (2018)
------------------------

Simulation code designed to reproduce SHG images of the colagen fibril network in the extra cellular matrix (ECM). Details on the model used can be found in documentation.pdf.

Installation:
-------------

``python3 setup.py install``


Instructions:
-------------

Main program and analysis can be run via the following commands:

1) ``python3 src/main.py [flags]``

	Flags are optional and can be entered in the command line or selected later during the setup process. 
	All input and output files will be created in the user's current working directory.

		-pos		Name of input position file 
				(will create a new one if does not exist)
		-param		Name of parameter file 
				(will create a new one if does not exist)
		-out		Name of output files (restart and trajectory)
		-ndim		Number of dimensions (must be 2 or 3)
		-nstep		Number of timesteps in simulation (default=10000)
		-vdw_sigma	Van de Waals radius of collagen beads in red. units
		-vdw_epsilon	Van de Waals energy in red. units
		-bond_k		Harmonic bond energy in red. units
		-angle_k	Sigmoidal angle energy in red. units
		-kBT		Temperature constant in red. units
		-Langevin	Whether to use Langevin dynamics (Y/N)
		-thermo_gamma	Value of Langevin collision rate gamma in red. units
		-nfibrilx	Number of collagen fibrils accross x axis
		-nfibrily	Number of collagen fibrils accross y axis
		-nfibrilz	Number of collagen fibrils accross z axis (3D only)
		-lfibril	Length of collagen fibrils in beads

2) ``python3 src/analysis.py [flags]``

	Flags are optional and can be entered in the command line or selected later during the analysis process.

		-traj	Name of trajectory file to analyse
		-param	Name of parameter file
		-gif	Name of gif file to be created
		-res	Image resolution parameter (1-10)
		-sharp	Image sharpness parameter (1-10)
		-skip	Number of sampled frames between each png
