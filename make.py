import sys, os, subprocess, multiprocessing

task = sys.argv[1]
program_name = sys.argv[2]
speed_test_name = "speed_test"
bin_dir = sys.argv[3]
python_command = 'python'
python_version = sys.version_info
ColECM_dir = os.getcwd()

if task in ['install', 'install_mpi']:

	print("Checking python executable version\n")

	if python_version[0] < 3:
		print("Error: current python version = {}.{}.{}\n       version required >= 3.0\n".format(python_version[0], python_version[1], python_version[2]))

		print("Checking python3 executable version\n")
		bashCommand = "python3 --version"
		try: process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		except:
			print("No python3 executable found, exiting installation\n\nYou need a Python 3 distribution to continue\n")
			sys.exit(1)

		output, error = process.communicate()
		python_version = output[:-1]

		if output[1] >= 3:	
			python_command += '3'
			print("{} detected, using python3 excecutable\n".format(output[:-1]))
		else: 
			print("No python3 excecutable found, exiting installation\n")
			sys.exit(1)

	print("Creating {} executable\n".format(program_name))

	if not os.path.exists(ColECM_dir + '/bin'): os.mkdir(ColECM_dir + '/bin')

	if task == 'install':
		with open(ColECM_dir + '/bin/' + program_name, 'w') as outfile:
			outfile.write('#!/bin/bash\n\n')
			outfile.write('{} {}/src/main.py "$@"'.format(python_command, ColECM_dir))

	else:
		with open(ColECM_dir + '/bin/' + program_name, 'w') as outfile:
			outfile.write('#!/bin/bash\n\n')
			outfile.write('{} {}/src/main_mpi.py "$@"'.format(python_command, ColECM_dir))

		with open(ColECM_dir + '/bin/' + speed_test_name, 'w') as outfile:
			outfile.write('#!/bin/bash\n\n')
			outfile.write('n=1\nnproc={}\n'.format(multiprocessing.cpu_count()))
			outfile.write('{} {}/src/main.py speed "$@"\n'.format(python_command, ColECM_dir))
			outfile.write('\nwhile [ $n -lt $nproc ]\n 	do\n')
			outfile.write('		mpirun -n $n {} {}/src/main_mpi.py speed "$@"\n'.format(python_command, ColECM_dir))
			outfile.write('		((n++))\n')
			outfile.write('		done\n')

		bashCommand = "chmod +x {}".format(ColECM_dir + '/bin/' + speed_test_name)
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()

	bashCommand = "chmod +x {}".format(ColECM_dir + '/bin/' + program_name)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()

	print("Copying {} executable to {}\n".format(program_name, bin_dir))

	bashCommand = "cp {} {}".format(ColECM_dir + '/bin/' + program_name, bin_dir)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()
	if error:
		print('!' * 30 + ' ERROR ' + '!' * 30 + '\n') 
		print("{}\nUnable to add ColECM to {}\n\nPlease manually create an alias to {}/{}\n".format(error, bin_dir, ColECM_dir, program_name))
		print('!' * 67 + '\n')


if task in ['uninstall', 'uninstall_mpi']:

	bashCommand = "rm {}".format(program_name)
	try: 
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()
	except: sys.exit(1)
	
	bashCommand = "rm {}/{}".format(bin_dir, program_name)
	try: 
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()
	except: sys.exit(1)
