import sys, os, subprocess

task = sys.argv[1]
program_name = sys.argv[2]
command = 'python'
python_version = sys.version_info
ColECM_dir = os.getcwd()

if task == 'install':

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
			command += '3'
			print("{} detected, using python3 excecutable\n".format(output[:-1]))
		else: 
			print("No python3 excecutable found, exiting installation\n")
			sys.exit(1)

	print("Creating {} executable\n".format(program_name))

	with open(program_name, 'w') as outfile:
		outfile.write('#!/bin/bash\n\n')
		outfile.write('{} {}/src/main.py "$@"'.format(command, ColECM_dir))

	bashCommand = "chmod +x {}".format(program_name)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()

	bashCommand = "which {}".format(command)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()
	bin_dir = str(output).split('/')[:-1]
	if bin_dir[0] == "b'": bin_dir = bin_dir[1:]
	bin_dir = '/' + "/".join(bin_dir)

	print("Copying {} executable to {}\n".format(program_name, bin_dir))

	bashCommand = "cp {} {}".format(program_name, bin_dir)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()
	if error:
		print('!' * 30 + ' ERROR ' + '!' * 30 + '\n') 
		print("{}\nUnable to add ColECM to {}\n\nPlease manually create an alias to {}/{}\n".format(error, bin_dir, ColECM_dir, program_name))
		print('!' * 67 + '\n')


if task == 'uninstall':

	bashCommand = "rm {}".format(program_name)
	try: 
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()
	except: sys.exit(1)
	
	bashCommand = "which {}".format(program_name)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()
	bin_dir = str(output).split('/')[:-1]
	if bin_dir[0] == "b'": bin_dir = bin_dir[1:]
	bin_dir = '/' + "/".join(bin_dir)

	bashCommand = "rm {}/{}".format(bin_dir, program_name)
	try: 
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()
	except: sys.exit(1)
