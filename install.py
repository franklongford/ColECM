import sys, os, subprocess

program_name = sys.argv[1]
command = 'python'
python_version = sys.version_info
ColECM_dir = os.getcwd()

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

with open(program_name, 'w') as outfile:
	outfile.write('#!/bin/bash\n\n')
	outfile.write('{} {}/src/main.py "$@"'.format(command, ColECM_dir))
