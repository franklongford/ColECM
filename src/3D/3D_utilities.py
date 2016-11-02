import numpy as np
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

def setup(N, L):

	POS_BEADS = np.zeros((N, 3))

	for i in xrange(N):
		for c in xrange(3):
			P_BEADS[i][c] = np.random.random() * L
			BEADS_HIS[i][0][c] = P_BEADS[i][c]
		for j in xrange(i):
			dx = (P_BEADS[i][0] - P_BEADS[j][0])
			dx -= L * int(2*dx/L)
			dy = (P_BEADS[i][1] - P_BEADS[j][1])
			dy -= L * int(2*dy/L)
			dz = (P_BEADS[i][2] - P_BEADS[j][2])
			dz -= L * int(2*dz/L)
			while (dx**2 + dy**2 + dz**2) < (2.5*sig1)**2:
				for c in xrange(3):
					P_BEADS[i][c] = np.random.random() * L
					BEADS_HIS[i][0][c] = P_BEADS[i][c]

				dx = (P_BEADS[i][0] - P_BEADS[j][0])
				dx -= L * int(2*dx/L)
				dy = (P_BEADS[i][1] - P_BEADS[j][1])
				dy -= L * int(2*dy/L)
				dz = (P_BEADS[i][2] - P_BEADS[j][2])
				dz -= L * int(2*dz/L)

	VEL_BEADS = (np.random.random((N,3)) - 0.5) * 2
	FRC_BEADS = np.zeros((N,3))

	return POS_BEADS, VEL_BEADS, FRC_BEADS


def calc_forces(N, P_BEADS, sig1, ep1):

	f_beads = np.zeros((N, 3))

	for i in xrange(N):
		for j in xrange(i):
			x = (P_BEADS[i][0] - P_BEADS[j][0])
			x -= L * int(2*x/L)
			y = (P_BEADS[i][1] - P_BEADS[j][1])
			y -= L * int(2*y/L)
			z = (P_BEADS[i][2] - P_BEADS[j][2])
			z -= L * int(2*z/L)

			"""
			r = np.sqrt(x**2 + y**2)
			f_beads[i][0] += x / r * FORCE(r, sig1, ep1)
			f_beads[i][1] += y / r * FORCE(r, sig1, ep1)
			"""

			r2 = x**2 + y**2 + z**2
			Fr = FORCE2(r2, sig1, ep1)
			f_beads[i][0] += x / r2 * Fr
			f_beads[i][1] += y / r2 * Fr
			f_beads[i][2] += z / r2 * Fr

			f_beads[j][0] -= x / r2 * Fr
			f_beads[j][1] -= y / r2 * Fr
			f_beads[j][2] += z / r2 * Fr

			#print "{} {} {}".format(x, y, r)

	return f_beads


def FORCE(r, sig1, ep1):

	return 24 * ep1 * (2 * (sig1/r)**12 - (sig1/r)**6) / r


def POT(r, sig1, ep1):

	return 4 * ep1 * ((sig1/r)**12 - (sig1/r)**6)

def POT2(r2, sig1, ep1):

	return 4 * ep1 * ((sig1**12/r2**6) - (sig1**6/r2**3))

def update_lines(num, dataLines, lines) :

    for line, data in zip(lines, dataLines) :
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2,:num])
    return lines




