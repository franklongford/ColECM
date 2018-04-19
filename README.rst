==================================================
ColECM - Collagen Extracellular Matrix Simulation
==================================================

By Frank Longford (2018)
------------------------

Simulation code designed to reproduce second harmonic generation (SHG) images of the collagen fibril network in the extra cellular matrix (ECM). Details on the model used can be found in documentation.pdf.

Installation:
-------------

run ``make``

Note: system must have working versions of Python and pip for python >= 3.0. Installation will check for common executables (``python, python3``) - please edit ``PYTHON`` and ``PIP`` macros on Makefile if local installations are named differently.


Instructions:
-------------

Main program and analysis can be run via the following commands:

1) ``ColECM simulation [flags]``

	Runs the main simulation routine for collagen fibrils in the ECM. Each fibril is approximated to be 1 um long.
	All flags except ``-pos`` and ``-param`` are optional and can be entered in the command line or selected later during the setup process. 
	All input and output files will be created in the user's current working directory.

		-pos		Name of simulation input position file 
				(will create a new file ending in '_pos' if does not exist)
		-param		Name of simulation parameter file 
				(will create a new file ending in '_param' if does not exist)
		-traj		Name of simulation trajectory file
		-rst		Name of simulation restart file
		-out		Name of simulation output file
		-ndim		Number of dimensions (2 or 3)
		-nstep		Number of timesteps in simulation
		-save_step	Number of timesteps between each saved restart and trajectory file
		-mass		Mass of each of collagen bead in red. units
		-vdw_sigma	Van de Waals radius of collagen beads in red. units
		-vdw_epsilon	Van de Waals energy in red. units
		-bond_k		Harmonic bond energy in red. units
		-angle_k	Sigmoidal angle energy in red. units
		-kBT		Temperature constant in red. units
		-gamma		Value of Langevin collision rate gamma in red. units (0-1)
		-lfib		Length of collagen fibrils in beads
		-nfibx		Number of collagen fibrils accross x axis
		-nfiby		Number of collagen fibrils accross y axis
		-nfibz		Number of collagen fibrils accross z axis (3D only)
		

2) ``ColECM analysis [flags]``

	Anisotropy analysis
	Flags are optional and can be entered in the command line or selected later during the analysis process.

		-param	Name of simulation parameter file
		-traj	Name of simulation trajectory file
		-out	Name of simulation output file to analyse
		-gif	Name of gif file to be created
		-res	Image resolution parameter (1-10)
		-sharp	Image sharpness parameter (1-10)
		-skip	Number of sampled frames between each png

You can run both modules by calling ``ColECM simulation analysis [flags]``

Defaults
--------

Simulation defaults are listed below:

	-traj		Name of simulation input position file
	-rst		Name of simulation input position file
	-out		Name of simulation trajectory file
	-gif		Name of simulation trajectory file
	-ndim		2
	-nstep		10000
	-save_step	500
	-mass		1
	-vdw_sigma	1
	-vdw_epsilon	2
	-bond_k		10
	-angle_k	10
	-kBT		5
	-gamma		1
	-lfib		5
	-nfibx		2
	-nfiby		2
	-nfibz		1
	-res		5
	-sharp		3
	-skip		10


File Tree:
-------------

Output of main routine will produce following file tree structure in the current working directory:

::

    sim
    │
    ├── ..._param.pkl
    ├── ..._pos.npy
    ├── ..._traj.npy
    └── ..._out.npy
	
    fig
    │
    ├── ..._energy_time.png
    ├── ..._energy_hist.png
    ├── ..._temp_time.png
    ├── ..._temp_hist.png
    ├── ..._anis_time.png
    └── ..._anis_hist.png

    gif
    │
    ├── ..._SHG_....gif
    └── ..._SHG_..._ISM.png  


Uninstallation:
-------------

run ``make uninstall`` and ``make clean``


Examples:
--------

Below are some examples:

1)  ``ColECM simulation analysis -pos test_defaults -param test_defaults``

	Will run and analyse a 2D simulation using the default parameter settings, usually lasing 10-20 seconds depending on system architecture.

2)  ``ColECM simulation analysis -pos test_3D -param test_3D -ndim 3``

	Will run and analyse a 3D simulation using the default parameter settings, usually lasing 10-20 seconds depending on system architecture.

 
