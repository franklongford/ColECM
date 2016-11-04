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

POS_BEADS, VEL_BEADS, FRC_BEADS, BOND_LIST= ut.setup(boxl, nchain, lchain, sig1, ep1, r0, kB)

ut.plot_system(POS_BEADS, VEL_BEADS, FRC_BEADS, N, boxl, bsize)

print "\n"
print "POSITIONS"
print POS_BEADS
print "\n"

print "VELOCITIES"
print VEL_BEADS
print "\n"

print "FORCES"
print FRC_BEADS
print "\n"

print "\n"
print "BOND LIST"
print BOND_LIST
print "\n"

for n in xrange(nsteps):
	for i in xrange(N):
		
		vx = VEL_BEADS[i][0] + 0.5 * dt * FRC_BEADS[i][0]
		vy = VEL_BEADS[i][1] + 0.5 * dt * FRC_BEADS[i][1]
		vz = VEL_BEADS[i][2] + 0.5 * dt * FRC_BEADS[i][2]

		POS_BEADS[i][0] += dt * vx
		POS_BEADS[i][1] += dt * vy
		POS_BEADS[i][2] += dt * vy

		for j in xrange(3): POS_BEADS[i][j] += boxl * (1 - int((POS_BEADS[i][j] + boxl) / boxl))

		FRC_BEADS = ut.calc_forces(N, boxl, POS_BEADS, BOND_LIST, sig1, ep1, r0, kB)

		VEL_BEADS[i][0] = vx + 0.5 * dt * FRC_BEADS[i][0]
		VEL_BEADS[i][1] = vy + 0.5 * dt * FRC_BEADS[i][1]
		VEL_BEADS[i][2] = vy + 0.5 * dt * FRC_BEADS[i][2]

	ut.plot_system(POS_BEADS, VEL_BEADS, FRC_BEADS, N, boxl, bsize)
	
	print "STEP ", n

	print "\n"
	print "POSITIONS"
	print POS_BEADS
	print "\n"

	print "VELOCITIES"
	print VEL_BEADS
	print "\n"

	print "FORCES"
	print FRC_BEADS
	print "\n"

plt.show()