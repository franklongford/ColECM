import numpy as np
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import utilities_3D as ut

nsteps = 500
nchain = 1
lchain = 10
N = nchain * lchain
boxl = 50

sig1 = 2.
bsize = sig1 * 300
ep1 = 3.0
dt = 0.005

r0 = 2. **(1./6.) * sig1
kB = 50.

pos, vel, frc, bond = ut.setup(boxl, nchain, lchain, sig1, ep1, r0, kB)
ut.plot_system(pos, vel, frc, N, boxl, bsize)

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

print "\n"
print "BOND LIST"
print bond
print "\n"

for n in xrange(nsteps):
	pos, vel, frc = ut.VV_alg(pos, vel, frc, bond, dt, N, boxl, sig1, ep1, r0, kB)
	ut.plot_system(pos, vel, frc, N, boxl, bsize)
	
	print "STEP ", n

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

plt.show()
