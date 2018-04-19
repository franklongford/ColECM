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

	Runs the main simulation routine for collagen fibrils in the ECM. Defaults parameter settings are listed. Each fibril is approximated to be 1 um long.
	All flags except ``-pos`` and ``-param`` are optional and can be entered in the command line or selected later during the setup process. 
	All input and output files will be created in the user's current working directory.

		-pos		Name of simulation input position file 
				(will create a new file ending in '_pos' if does not exist)
		-param		Name of simulation parameter file 
				(will create a new file ending in '_param' if does not exist)
		-traj		Name of simulation trajectory file
		-rst		Name of simulation restart file
		-out		Name of simulation output file
		-ndim 2		Number of dimensions (2 or 3)
		-nstep 10000		Number of timesteps in simulation
		-save_step 500		Number of timesteps between each saved restart and trajectory file
		-mass 1		Mass of each of collagen bead in red. units
		-vdw_sigma 1	Van de Waals radius of collagen beads in red. units
		-vdw_epsilon 2	Van de Waals energy in red. units
		-bond_k 20		Harmonic bond energy in red. units
		-angle_k 20	Sigmoidal angle energy in red. units
		-kBT 5		Temperature constant in red. units
		-gamma 1		Value of Langevin collision rate gamma in red. units (0-1)
		-lfib 5		Length of collagen fibrils in beads
		-nfibx 2		Number of collagen fibrils accross x axis
		-nfiby 2		Number of collagen fibrils accross y axis
		-nfibz 1		Number of collagen fibrils accross z axis (3D only)
		

2) ``ColECM analysis [flags]``

	Anisotropy analysis
	Flags are optional and can be entered in the command line or selected later during the analysis process.

		-param	Name of simulation parameter file
		-traj	Name of simulation trajectory file
		-out	Name of simulation output file to analyse
		-gif	Name of gif file to be created
		-res 5	Image resolution parameter (1-10)
		-sharp 3	Image sharpness parameter (1-10)
		-skip 10	Number of sampled frames between each png

You can run both modules by calling ``ColECM simulation analysis [flags]``


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

 
