==================================================
ColECM - Collagen Extracellular Matrix Simulation
==================================================

By Frank Longford (2018)
------------------------

Simulation code designed to reproduce second harmonic generation (SHG) images of the collagen fibril network in the extra cellular matrix (ECM). Details on the model used can be found in documentation.pdf. Please download latest version released as master branch copy may be unstable.

Installation:
-------------

System must have working versions of Python and pip for python >= 3.0. Installation will check for common executables (``python, python3``) - please edit ``PYTHON`` and ``PIP`` macros on Makefile if local installations are named differently. Parallel version also requires Open MPI compilers.

Users are encouranged to use a ``conda`` environment in order to manage thier installation (either via ``anaconda`` or ``miniconda``).

1) enter the ``ColECM`` directory

2) type ``make``

For parallel version:

3) type ``make install_mpi``

NB - due to implementation, SERIAL version will ALWAYS BE FASTER than MPI running on 1 NODE

Instructions:
-------------

Main program and analysis can be run via the following commands:

1) ``ColECM simulation [flags]``  or  ``mpirun -n [nproc] ColECM_mpi simulation [flags]``

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
		-vdw_sigma	 Van de Waals radius of collagen beads in red. units
		-vdw_epsilon	 Van de Waals energy in red. units
		-bond_k0		Harmonic bond energy in red. units
		-angle_k0	Sigmoidal angle energy in red. units
		-kBT		Temperature constant in red. units
		-gamma		Value of Langevin collision rate gamma in red. units (0-1)
		-lfib		Length of collagen fibrils in beads
		-nfibx		Number of collagen fibrils accross x axis
		-nfiby		Number of collagen fibrils accross y axis
		-nfibz		Number of collagen fibrils accross z axis (3D only)
		

2) ``ColECM analysis [flags]`` or  ``mpirun -n [nproc] ColECM_mpi analysis [flags]``

	Anisotropy analysis
	Flags are optional and can be entered in the command line or selected later during the analysis process.

		-param	Name of simulation parameter file
		-traj	Name of simulation trajectory file
		-out	Name of simulation output file to analyse
		-gif	Name of gif file to be created
		-res	Image resolution parameter (1-10)
		-sharp	Image sharpness parameter (1-10)
		-skip	Number of sampled frames between each png


3) ``ColECM editor [flags]`` or  ``mpirun -n [nproc] ColECM_mpi editor [flags]``

	Simulation editor
	Parameter file can be edited using flags, as well as simulation cell expanded using repeated units in x, y and z dimensions.

		-param	Name of simulation parameter file
		-rst	Name of simulation restart file
		-nrepx	Number of simulation cells to repeat in x dimension
		-nrepy	Number of simulation cells to repeat in y dimension
		-nrepz	Number of simulation cells to repeat in z dimension (3D only)

	Generally it is not necessary to run MPI version of editor module, only when expanding simulation cell, since the temperature will be equilibrated afterwards to obtain an appropriate ensemble of velocities.

A speed test can be found in the binary folder to estimate the optimum number of processors to use for a MPI run. The number of processors available will be automatically detected, but can be changed in the ``speed_test`` executable

4) ``~/ColECM/bin/speed_test [flags]``

	Simulation editor
	Flags are optional and can be entered in the command line or selected later during the analysis process.

		-param	Name of simulation parameter file
		-rst	Name of simulation restart file
		-ntrial  Number of trial calculations to perform in speed test (default=1000)

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
	-vdw_epsilon	1
	-bond_r0	1.122
	-bond_k0	1
	-angle_k0	1
	-rc		3.0
	-kBT		1
	-gamma		0.5
	-lfib		10
	-nfibx		3
	-nfiby		3
	-nfibz		1
	-density	0.3
	-res		7.5
	-sharp		1
	-skip		1


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

run ``make uninstall`` or ``make uninstall_mpi`` and ``make clean``


Examples:
--------

Below are some examples:

1)  ``ColECM simulation analysis -pos test_defaults -param test_defaults``

	Will run and analyse a 2D simulation using the default parameter settings, usually lasing 10-20 seconds depending on system architecture.

2)  ``mpirun -n 4 ColECM_mpi simulation -pos test_defaults -param test_defaults``

	Will run a 2D simulation on 4 processors using the default parameter settings, usually lasing 10-20 seconds depending on system architecture.

3)  ``ColECM simulation analysis -pos test_3D -param test_3D -ndim 3``

	Will run and analyse a 3D simulation using the default parameter settings, usually lasing 10-20 seconds depending on system architecture.

4)  ``ColECM analysis -pos test_3D -param test_3D``

	Will analyse a 3D simulation as defined by position and parameter file names using the default parameter settings.

5)  ``ColECM analysis -pos test_3D -param test_3D -res 10 -sharp 4``

	Will analyse a 3D simulation as defined by position and parameter file names using increased image resolution and sharpness.

6)  ``ColECM editor -rst test_3D -param test_3D -nrepx 2 -nrepy 3``

	Will take in ``test_3D`` restart file any create a new system by repeating unit cell x2 in x dimension and x3 in y dimension.
