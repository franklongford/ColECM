import numpy as np
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

def unit_vector(vector):

	sum_ = np.sum([i**2 for i in vector])
	norm = 1./sum_
	u_vector = [np.sqrt(i**2 * norm) * np.sign(i) for i in vector]

	return u_vector


def rand_vector(n): return np.array(unit_vector(np.random.random((n)) * 2 - 1)) 


def setup(N, L, sig1):

	POS_BEADS = np.zeros((N, 3))
	POS_BEADS[0] += np.random.random((3)) * L 
	for i in xrange(N-1):
		next_bead = POS_BEADS[i] + rand_vector(3) * sig1
		for n in xrange(3): next bead[n] += L * (1 - int((next_bead[n] + L) / L))
		POS_BEADS[i+1] += next_bead

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




