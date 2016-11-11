import numpy as np
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import os
import sys
import utilities_3D as ut

nsteps = 500
rho = 3
nchain = 10
lchain = 20
N = nchain * lchain
boxl = (N / 3)**(1./3)

a_ii = 10
a_ij = 10
sigma = 3.67
bsize = sigma * 800
gamma = sigma**2 / 2
lam = 0.65
dt = 0.015
r_m = 1.25
K = 20.0
sqrt_dt = np.sqrt(dt)
root = '/data/fl7g13/Collagen/Output'

pos, vel, frc, bond, polar, A, chains, beads = ut.setup_DPD(boxl, nchain, lchain, [a_ii, a_ij], gamma, sigma, sqrt_dt, r_m, K)
#ut.plot_system_DPD(pos, vel, frc, N, boxl, bsize, TYPE)

print "\n"
print "POSITIONS"
print pos
print "\n"

print "VELOCITIES"
print vel
print "\n"

print "FORCES"
print frc
print "\n"

print "TYPE"
print TYPE
print "\n"

print "REPULSION"
print A
print "\n"

print "BOND"
print bond
print "\n"

print "POLAR"
print polar
print "\n"

pos_history = np.zeros((nsteps, N, 3))
vel_history = np.zeros((nsteps, N, 3))
frc_history = np.zeros((nsteps, N, 3))
bond_history = np.zeros((nsteps, N, N))
TYPE_history = np.zeros((nsteps, N))

for n in xrange(nsteps):
	pos, vel, frc = ut.VV_alg_DPD(pos, vel, frc, dt, N, boxl, bond, polar, A, gamma, sigma, lam, sqrt_dt, r_m, K, lchain, TYPE)
	pos_history[n] += pos
	vel_history[n] += vel
	frc_history[n] += frc
	bond_history[n] += bond
	TYPE_history[n] += TYPE
	ut.save_system(root, pos_history, vel_history, frc_history, bond_history, TYPE_history, nchain, lchain)
		
	sys.stdout.write("STEP {}\r".format(n) )
	sys.stdout.flush()

	"""
	print "\n"
	print "POSITIONS"
	print pos
	print "\n"

	print "VELOCITIES"
	print vel
	print "\n"

	print "FORCES"
	print frc
	print "\n"
	"""

	print "TYPE"
	print TYPE
	print "\n"

	print "REPULSION"
	print A
	print "\n"

	print "BOND"
	print bond
	print "\n"

	print "POLAR"
	print polar
	print "\n"

pos_history, vel_history, frc_history, bond_history, TYPE_history = ut.load_system(root, nchain, lchain)
for n in xrange(nsteps):
	ut.plot_system_DPD(pos_history[n], vel_history[n], frc_history[n], bond_history[n], N, boxl, bsize, TYPE_history[n], n)

