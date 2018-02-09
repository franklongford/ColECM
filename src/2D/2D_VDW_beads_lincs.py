"""
COLLAGEN FIBRIL SIMULATION 2D

Created by: Frank Longford
Created on: 01/11/15

Last Modified: 24/11/15
"""


import numpy as np
import scipy.constants as con
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os



def make_chains(Nch, Nbe, L, sig1, r0, theta0):
	"Creates Nch fibrils chains containing Nbe beads in a cell with an area of L x L"

	facb = 0.1
	faca = 0.1

	dL = float(L) / (Nch + 1)

	P_BEADS = np.zeros((Nch*Nbe, 2))
	B_BEADS = np.zeros((Nch*Nbe, Nch*Nbe))
	NB_BEADS = np.ones((Nch*Nbe, Nch*Nbe))
	con_index= np.zeros((Nch*(Nbe-1), cmax))
	con_coeff= np.zeros((Nch*(Nbe-1), cmax))
	atom1 = np.zeros(Nch*(Nbe-1))
	atom2 = np.zeros(Nch*(Nbe-1))
	ncc = np.zeros(K, dtype=int)
	kcount = 0

	for i in xrange(Nch*Nbe):

		NB_BEADS[i][i] = 0

		if np.remainder(i, Nbe) == 0:

			P_BEADS[i][0] = (np.int(float(i) / Nbe) + 1) * dL 
			P_BEADS[i][1] = np.random.random() * L

			#print i, np.remainder(i, Nbe), (int(float(i) / Nbe) + 1)

		else:

			r = r0
			if TYPE_B == 0: r += np.random.random() * 2 * facb - facb
			theta = theta0 + np.random.random() * 2 * faca - faca

			P_BEADS[i][0] = P_BEADS[i-1][0] + r * np.sin(theta) 
			P_BEADS[i][1] = P_BEADS[i-1][1] + r * np.cos(theta)

			B_BEADS[i][i-1] = 1
			B_BEADS[i-1][i] = 1
			NB_BEADS[i][i-1] = 0
			NB_BEADS[i-1][i] = 0
			atom1[kcount] = i-1
			atom2[kcount] = i
			kcount += 1
	
			for k in xrange(2):
				if P_BEADS[i][k] > L: P_BEADS[i][k] -= int(P_BEADS[i][k] / L) * L
				elif P_BEADS[i][k] < 0: P_BEADS[i][k] -= int(P_BEADS[i][k] / L - 1) * L

			#if np.remainder(i, Nbe) != 0 and np.remainder(i, Nbe) != Nbe - 1: DA_BEADS[i] = 1


	"""
	for i in xrange(Nch):
		for j in xrange(Nbe-1):
			ii = i * (Nbe - 1)
			jj = i * (Nbe - 1) + j
			for k in xrange(cmax):
				#print i, j, jj, k,  j+ k, np.remainder(j + k + 1, Nbe-1) 
				con_index[jj][k] = ii + np.remainder(j + k + 1, Nbe-1)
	"""

	for i in xrange(Nch*(Nbe-1)):

		if np.remainder(i, Nbe-1) == 0: 
			ncc[i] = int(1)
			con_index[i][0] = i + 1
		elif np.remainder(i, Nbe-1) == Nbe - 2: 
			ncc[i] = int(1)
			con_index[i][0] = i - 1
		else: 
			ncc[i] = int(2)
			con_index[i][0] = i - 1
			con_index[i][1] = i + 1

		for j in xrange(ncc[i]):
			print i, j, atom1[i], atom1[con_index[i][j]], atom2[i], atom2[con_index[i][j]]
			if atom1[i] == atom1[con_index[i][j]] or atom2[i] == atom2[con_index[i][j]]:  
				con_coeff[i][j] = - 1. / m * S**2
			else:
				con_coeff[i][j] = 1. / m * S**2


	return P_BEADS, B_BEADS, NB_BEADS, ncc, con_index, con_coeff, atom1, atom2 #DA_BEADS


def grow_chains(P_BEADS, V_BEADS, F_BEADS, R_BONDS, sig1, r0, theta0):

	facb = 0.2
	faca = 1.5

	for i in xrange(Nch):
		if np.random.random() >= 0.9:
			for j in xrange(Nch):
				for k in xrange(Nbe):
					dx = (P_BEADS[i][-1][0] - P_BEADS[j][k][0])
					dx -= L * int(2*dx/L)
					dy = (P_BEADS[i][-1][1] - P_BEADS[j][k][1])
					dy -= L * int(2*dy/L)

					if (dx**2 + dy**2) < (3*sig1)**2: break
			x = P_BEADS[i][-1][0] + r0 * np.sin(theta0) 
			y = P_BEADS[i][-1][1] + r0 * np.cos(theta0)
			P_BEADS[i].np.append([x, y])


def add_globule(P_GLOB, L, sig2):

	gx = np.random.random() * L
	gy = np.random.random() * L

	P_GLOB.append([gx, gy, sig2])


def forces(P_BEADS, P_GLOB, Nch, Nbe, L, sig1, r0, theta0):


	F_BEADS = np.zeros((Nch*Nbe, 2))
	R_BONDS = np.zeros((Nch*(Nbe-1), 3))

	k = 0

	for i in xrange(Nch*Nbe): 

		for j in xrange(i):

			dx = (P_BEADS[j][0] - P_BEADS[i][0])
			dx -= L * int(2*dx/L)
			dy = (P_BEADS[j][1] - P_BEADS[i][1])
			dy -= L * int(2*dy/L)

			r2 = dx**2 + dy**2

			if B_BEADS[i][j] == 1:

				r = np.sqrt(r2)

				R_BONDS[k][0] = dx
				R_BONDS[k][1] = dy
				R_BONDS[k][2] = r

				k += 1

				FB = FORCE_BOND(r, r0, kB)

				F_BEADS[j][0] += dx / r * FB
				F_BEADS[j][1] += dy / r * FB

				F_BEADS[i][0] -= dx / r * FB
				F_BEADS[i][1] -= dy / r * FB


			FVDW = FORCE_VDW2(r2, sig1, ep1) * NB_BEADS[i][j] * TYPE_VDW

			F_BEADS[j][0] += dx / r2 * FVDW
			F_BEADS[j][1] += dy / r2 * FVDW

			F_BEADS[i][0] -= dx / r2 * FVDW
			F_BEADS[i][1] -= dy / r2 * FVDW

		if TYPE_G == 1:
			for j in xrange(len(P_GLOB)):

				dx = (P_BEADS[i][0] - P_GLOB[j][0])
				dx -= L * int(2*dx/L)
				dy = (P_BEADS[i][1] - P_GLOB[j][1])
				dy -= L * int(2*dy/L)

				r2 = dx**2 + dy**2

				FG = FORCE_GLOB(r2, P_GLOB[j][2])

				F_BEADS[i][0] += dx / r2 * FG
				F_BEADS[i][1] += dy / r2 * FG

				#print i, P_GLOB[j][2], np.sqrt(r2), dx / r2 * FG, dy / r2 * FG, FG / np.sqrt(r2), np.sqrt((dx / r2 * FG)**2 + (dy / r2 * FG)**2)

	
	if TYPE_A == 1:

		for i in xrange(Nch):
			for j in xrange(Nbe-2):

				ii = i * Nbe + j + 1
				jj = i * (Nbe - 1) + j

				Rijx = R_BONDS[jj][0]
				Rijy = R_BONDS[jj][1]
				rij = R_BONDS[jj][2]

				Rjkx = R_BONDS[jj+1][0]
				Rjky = R_BONDS[jj+1][1]
				rjk = R_BONDS[jj+1][2]

				dot_prod = (Rijx * Rjkx) + (Rjky * Rjky)

				factor = 1. / (rij * rjk)
				#factor = 2. * (dot_prod / (rij * rjk)**2  + 1. / (rij * rjk)) 

				F_BEADS[ii-1][0] += ka * (Rjkx - Rijx * dot_prod / rij**2) * factor
				F_BEADS[ii-1][1] += ka * (Rjky - Rijy * dot_prod / rij**2) * factor

				F_BEADS[ii][0] += ka * (Rijx - Rjkx + Rijx * dot_prod / rij**2 - Rjkx * dot_prod / rjk**2) * factor
				F_BEADS[ii][1] += ka * (Rijy - Rjky + Rijy * dot_prod / rij**2 - Rjky * dot_prod / rjk**2) * factor

				F_BEADS[ii+1][0] -= ka * (Rijx - Rjkx * dot_prod / rjk**2) * factor
				F_BEADS[ii+1][1] -= ka * (Rijy - Rjky * dot_prod / rjk**2) * factor


	return F_BEADS, R_BONDS



def shake(Vm1, P_BEADS, R_BONDS):


	for i in xrange(Nch):
		check = np.ones((Nbe-1))
		while np.sum(check) != 0:

			for j in xrange(Nbe-1):

				#print i, j, i * Nbe + j,  i * (Nbe - 1) + j

				ii = i * Nbe + j
				jj = i * (Nbe - 1) + j

				sx = (P_BEADS[ii][0] + dt * Vm1[ii][0] - P_BEADS[ii+1][0] - dt * Vm1[ii+1][0])
				sx -= L * int(2*sx/L)
				sy = (P_BEADS[ii][1] + dt * Vm1[ii][1] - P_BEADS[ii+1][1] - dt * Vm1[ii+1][1])
				sy -= L * int(2*sy/L)
				
				s2 = sx**2 + sy**2

				if abs(s2 - r0**2) >= 0.1 * r0**2:

					if abs(s2 - r0**2) > 5000 * r0:
						print "SHAKE BOMB nstep = {} {} {} {}".format( n, j, j+1, abs(s2 - r0**2))
						#print Vm1
						#return
						sys.exit()


					#print "SHAKE {} {}".format(s2 - r0**2 , 0.1 * r0**2)

					check[j] = 1

					tx = R_BONDS[jj][0]
					ty = R_BONDS[jj][1]
					tr = R_BONDS[jj][2]

					g = (s2 - r0**2) / (4 * m * dt * ((sx * tx) + (sy * ty)))

					Vm1[ii][0] -= g * tx / m
					Vm1[ii][1] -= g * ty / m

					Vm1[ii+1][0] += g * tx / m
					Vm1[ii+1][1] += g * ty / m

				else: check[j] = 0

	for i in xrange(Nch*Nbe):
		for k in xrange(2):
			P_BEADS[i][k] += dt * Vm1[i][k]

			if P_BEADS[i][k] > L: P_BEADS[i][k] -= int(P_BEADS[i][k] / L) * L
			elif P_BEADS[i][k] < 0: P_BEADS[i][k] -= int((P_BEADS[i][k] / L) - 1) * L


def rattle(Vm1, V_BEADS, R_BONDS):


	for i in xrange(Nch):

		check = np.ones(Nbe-1)

		while np.sum(check) != 0:

			for j in xrange(Nbe-1):

				ii = i * Nbe + j
				jj = i * (Nbe - 1) + j

				dvx = (V_BEADS[ii][0] - V_BEADS[ii+1][0])
				dvy = (V_BEADS[ii][1] - V_BEADS[ii+1][1])

				dot_prod = (dvx * R_BONDS[jj][0]) + (dvy * R_BONDS[jj][1])

				if abs(dot_prod) >= 0.05:

					if abs(dot_prod) > 5000 * r0:
						print "RATTLE BOMB nstep = {} {} {} {}".format( n, j, j+1, dot_prod)
						#print V_BEADS
						#return
						sys.exit()

					check[j] = 1

					#print "RATTLE {}".format(dot_prod)

					k = dot_prod / (2 * m * r0**2)

					V_BEADS[ii][0] -= k * R_BONDS[jj][0] / m
					V_BEADS[ii][1] -= k * R_BONDS[jj][1] / m

					V_BEADS[ii+1][0] += k * R_BONDS[jj][0] / m
					V_BEADS[ii+1][1] += k * R_BONDS[jj][1] / m

				else: check[j] = 0



def gauss(sigma, l0=0.):

	r = 2.0
	while r >= 1.0:
		v1 = 2 * np.random.random() - 1
		v2 = 2 * np.random.random() - 1
		r = v1**2 + v2**2
	l = v1 * np.sqrt(-2 * np.log(r) / r)
	l = l0 + sigma * l
	return l


def FORCE_BOND(r, r0, kB):

	return 2 * kB * (r0 - r)

def FORCE_ANGLE(the, the0, ka):

	return 2 * ka * (the0 - the)


def FORCE_VDW(r, sig1, ep1):

	return 24 * ep1 * (2 * (sig1/r)**12 - (sig1/r)**6) / r


def FORCE_VDW2(r2, sig1, ep1):

	return 24 * ep1 * (2 * (sig1**12/r2**6) - (sig1**6/r2**3))

def FORCE_GLOB(r2, sig2):

	return 12 * ep2 *((sig2**12/r2**6))

def FORCE_GLOB_2(r, sig2, ep2):

	return 12 * ep2 * ((sig2**12/r**13))


def POT_VDW(r, sig1, ep1):

	return 4 * ep1 * ((sig1/r)**12 - (sig1/r)**6)

def POT_BOND(r, r0, kB):

	return kB * (r - r0)**2

def POT_ANGLE(the, the0, ka):
	
	return ka * (the - the0)**2

def POT_INTRA(r2, sig1, ep1):

	return ep1 * ((sig1/r)**2)

def POT_GLOB(r, sig2, ep2):

	return ep2 * (sig2/r)**12

def import_restart(name):

	FILE = open(name, 'r')
	IN = FILE.readlines()
	FILE.close()

	temp_lines = IN[0].split() 
	Nch = int(temp_lines[0])
	Nbe = int(temp_lines[1])

	P_BEADS = np.zeros((Nch*Nbe, 2))
	V_BEADS = np.zeros((Nch*Nbe, 2))

	B_BEADS = np.zeros((Nch*Nbe, Nch*Nbe))
	NB_BEADS = np.ones((Nch*Nbe, Nch*Nbe))
	DA_BEADS = np.zeros((Nch*Nbe))

	P_GLOB = ([])

	for i in xrange(Nch*Nbe):
		NB_BEADS[i][i] = 0

		if np.remainder(i, Nbe) != 0:
			B_BEADS[i][i-1] = 1
			B_BEADS[i-1][i] = 1
			NB_BEADS[i][i-1] = 0
			NB_BEADS[i-1][i] = 0

		if np.remainder(i, Nbe) != 0 and np.remainder(i, Nbe) != Nbe - 1: DA_BEADS[i] = 1

		temp_lines = IN[i+1].split()
		P_BEADS[i][0] = float(temp_lines[0])
		P_BEADS[i][1] = float(temp_lines[1])
		V_BEADS[i][0] = float(temp_lines[2])
		V_BEADS[i][1] = float(temp_lines[3])

	for i in xrange(len(IN))[Nch*Nbe][-1]:
		temp_lines = IN[i].split()
		P_GLOB.append([float(temp_lines[0]), float(temp_lines[1]), float(temp_lines[2])])


	return P_BEADS, V_BEADS, B_BEADS, NB_BEADS, DA_BEADS, P_GLOB


def save_restart(P_BEADS, V_BEADS, P_GLOB, Nch, Nbe, T, L, nsteps, n, PATH):

	FILE = open("{}/Restarts/restart_2D_{}_{}_{}_{}_{}_{}.txt".format(PATH, Nch, Nbe, int(L), n, int(ka), int(T)), 'w')

	FILE.write("{} {} {} {} {}\n".format(Nch, Nbe, T, L, nsteps))

	for i in xrange(Nch*Nbe):
		FILE.write("{} {} {} {}\n".format(P_BEADS[i][0], P_BEADS[i][1], V_BEADS[i][0], V_BEADS[i][1]))

	for i in xrange(len(P_GLOB)):
		FILE.write("{} {} {}\n".format(P_GLOB[i][0], P_GLOB[i][1], P_GLOB[i][2]))

	FILE.close()

#"""

def solve(nrec, con_index, con_coeff, xp, atom1, atom2, K, B, A, rhs, sol):

	w = 1

	for rec in xrange(nrec):
		for i in xrange(K):
			rhs[w][i] = 0
			for j in xrange(ncc[i]):
				rhs[w][i] += A[i][j] * rhs[1-w][con_index[i][j]]
			sol[i] += rhs[w][i]
		w = 1 - w

	for i in xrange(K):
		a1 = atom1[i]
		a2 = atom2[i]

		for k in xrange(2):
			xp[a1][k] -= 1./ m * B[i][k] * S * sol[i]
			xp[a2][k] += 1./ m * B[i][k] * S * sol[i]
			

def lincs(P_BEADS, V_BEADS, R_BONDS, con_index, con_coeff, atom1, atom2, nrec=3):

	#print P_BEADS

	K = Nch * (Nbe - 1)

	xp = np.zeros((Nch * Nbe, 2))
	tmp = np.zeros((Nch * Nbe, 2))
	A = np.zeros((K,cmax))
	B = np.zeros((K, 2))
	rhs = np.zeros((2, K))
	sol = np.zeros(K)

	"Update positions and velocities based on forces"
	for i in xrange(Nch*Nbe):
		for k in xrange(2):
			xp[i][k] = P_BEADS[i][k] + dt * Vm1[i][k]
			tmp[i][k] = P_BEADS[i][k] + dt * Vm1[i][k]

	for i in xrange(K):
		for k in xrange(2):
			B[i][k] =  R_BONDS[i][k] / R_BONDS[i][2]

	for i in xrange(K):
		for n in xrange(ncc[i]):
			k = con_index[i][n]

			A[i][n] = con_coeff[i][n] * (B[i][0] * B[k][0] + B[i][1] * B[k][1])

		a1 = atom1[i]
		a2 = atom2[i]

		dx = (xp[a1][0] - xp[a2][0])
		dx -= L * int(2*dx/L)
		dy = (xp[a1][1] - xp[a2][1])
		dy -= L * int(2*dy/L)

		rhs[0][i] = S * (B[i][0] * dx + B[i][1] * dy - r0 )

		sol[i] = rhs[0][i]  

	solve(nrec, con_index, con_coeff, xp, atom1, atom2, K, B, A, rhs, sol)

	for i in xrange(K):

		a1 = atom1[i]
		a2 = atom2[i]

		dx = (xp[a1][0]  - xp[a2][0])
		dx -= L * int(2*dx/L)
		dy = (xp[a1][1]  - xp[a2][1])
		dy -= L * int(2*dy/L)

		print a1, a2, 2 * r0**2 - dx**2 - dy**2, dx**2 + dy**2, 2 * r0**2

		p = np.sqrt(2 * r0**2 - dx**2 - dy**2 )

		rhs[0][i] = S * (r0 - p)

		sol[i] = rhs[0][i]

	solve(nrec, con_index, con_coeff, xp, atom1, atom2, K, B, A, rhs, sol)

	print tmp - xp

	for i in xrange(Nch*Nbe):
		for k in xrange(2):
			P_BEADS[i][k] = xp[i][k]

			if P_BEADS[i][k] > L: P_BEADS[i][k] -= int(P_BEADS[i][k] / L) * L
			elif P_BEADS[i][k] < 0: P_BEADS[i][k] -= int((P_BEADS[i][k] / L) - 1) * L

	#print P_BEADS
			
#"""

TYPE_B = 0
TYPE_A = 0
TYPE_VDW = 1
TYPE_T = 1
TYPE_G = 1

nsteps = 1000
Nch = 10
Nbe = 12

K = Nch * (Nbe - 1)
cmax = 2

m = 1.0
M = np.ones((2*Nch*Nbe,2*Nch*Nbe)) * m
S = np.sqrt(m / 2.)

L = 22.
T = 2.0

sig1 = 1.
bsize = sig1 * 100
ep1 = 1.0

Pglob = 0.005
cglob = 0.03
maxglobs = 50.0
maxglobn = 1
sig2 = 1.0
ep2 = 50.0

"""
Y = map (lambda x: FORCE_GLOB(x**2, sig2) / x, np.linspace(0.01, 10, 100))
print Y
plt.plot(np.linspace(0.01, 10, 100), Y)
plt.axis([0,10,0,5])
plt.show()
plt.close()
"""

dt = 0.02
nu = 5.0

r0 = 2. **(1./6.) * sig1
theta0 = np.pi
kB = 50.
ka = 50.

origres = 0
nres = 50
folder = 7

PATH = "Beads_{}".format(folder)

if os.path.isdir(PATH) == False: 
	os.mkdir(PATH)
	os.mkdir("{}/Restarts".format(PATH))
	os.mkdir("{}/Figures".format(PATH))

FILE = "{}/Restarts/restart_2D_{}_{}_{}_{}_{}_{}.txt".format(PATH, Nch, Nbe, int(L), origres, int(ka), int(T))
if os.path.isfile(FILE) == 1 and origres > 0:
	P_BEADS, V_BEADS, B_BEADS, NB_BEADS, DA_BEADS, P_GLOB = import_restart(FILE)
	print "IMPORTED RESTART"
	res = origres + 1

else:
	P_BEADS, B_BEADS, NB_BEADS, ncc, con_index, con_coeff, atom1, atom2 = make_chains(Nch, Nbe, L, sig1, r0, theta0)
	print ncc
	print con_index
	print con_coeff

	V_BEADS = np.random.random((Nch*Nbe, 2)) - 0.5
	sumv1 = np.sum(V_BEADS)
	sumv1 = sumv1 / (Nch * Nbe)
	sumv2 = 0.
	for i in xrange(Nch*Nbe):
		for k in xrange(2):
			sumv2 += V_BEADS[i][k]**2
	sumv2 = sumv2 / (Nch * Nbe)
	fs = np.sqrt(3 * T / sumv2)

	for i in xrange(Nch*Nbe):
		for k in xrange(2):
			V_BEADS[i][k] = (V_BEADS[i][k] - sumv1) * fs

	P_GLOB = ([])
	if TYPE_G == 1:
		add_globule(P_GLOB, L, sig2)
		#if np.random.random() <= Pglob: add_globule(P_GLOB, L, sig2)

	#save_restart(P_BEADS, V_BEADS, P_GLOB, Nch, Nbe, T, L, nsteps, origres)

	fig = plt.figure(0)
	col = 0
	for i in xrange(Nch):
		plotxy = np.rot90(P_BEADS)
		plt.scatter(plotxy[0], plotxy[1], s=bsize)
		col += 1
	if len(P_GLOB) != 0:
			for g in xrange(len(P_GLOB)):
				plt.scatter(P_GLOB[g][0], P_GLOB[g][1], s=P_GLOB[g][2]*50, c='red')
	plt.axis([0,L,0,L])
	plt.savefig('{}/Figures/{}.png'.format(PATH, 0))
	res = 0

F_BEADS = np.zeros((Nch*Nbe, 2))
F_BEADS, R_BONDS = forces(P_BEADS, P_GLOB, Nch, Nbe, L, sig1, r0, theta0)

print "\n"
print "POSITIONS"
print P_BEADS
print "\n"

print "\n"
print "BONDS"
print B_BEADS
print "\n"

print "\n"
print "NON-BONDS"
print NB_BEADS
print "\n"

"""
print "\n"
print "BOND ANGLES"
print DA_BEADS
print "\n"
"""

print "VELOCITIES"
print V_BEADS
print "\n"

print "FORCES"
print F_BEADS
print "\n"

print "BOND LENGTHS"
print R_BONDS
print "\n"

sigma = np.sqrt(T)
Tav = 0

for n in xrange(nsteps):

	Vm1 = np.zeros((Nch*Nbe, 2))

	#P_BEADS_copy = np.zeros((Nch, Nbe, 2))

	for i in xrange(Nch*Nbe):

		for k in xrange(2):

			Vm1[i][k] = V_BEADS[i][k] + 0.5 * dt * F_BEADS[i][k] / m

			if TYPE_B == 0: 
				P_BEADS[i][k] += dt * Vm1[i][k]

				if P_BEADS[i][k] > L: P_BEADS[i][k] -= int(P_BEADS[i][k] / L) * L
				elif P_BEADS[i][k] < 0: P_BEADS[i][k] -= int((P_BEADS[i][k] / L) - 1) * L

			#if Pm1[i][j][k] > L: Pm1[i][j][k] -= int(Pm1[i][j][k] / L) * L
			#elif Pm1[i][j][k] < 0: Pm1[i][j][k] -= int((Pm1[i][j][k] / L) - 1) * L

	if TYPE_B == 1: shake(Vm1, P_BEADS, R_BONDS)
	elif TYPE_B == 2: lincs(P_BEADS, V_BEADS, R_BONDS, con_index, con_coeff, atom1, atom2, 3)

	F_BEADS, R_BONDS = forces(P_BEADS, P_GLOB, Nch, Nbe, L, sig1, r0, theta0)

	fig = plt.figure(n+1)
	col = 0
	tempa = 0.

	for i in xrange(Nch*Nbe):
		for k in xrange(2):
			V_BEADS[i][k] = Vm1[i][k] + 0.5 * dt * F_BEADS[i][k] / m
			tempa += V_BEADS[i][k]**2

	if TYPE_B == 1: rattle(Vm1, V_BEADS, R_BONDS, i)

	tempa = tempa / (2 * Nch * Nbe)
	Tav += tempa / nsteps
	#if TYPE_T == 1: sigma = np.sqrt(T)

	for i in xrange(Nch*Nbe):
		if TYPE_T == 1:
			if np.random.random() <= nu * dt:
				for k in xrange(2):
					V_BEADS[i][k] = gauss(sigma, 0.)


		plotxy = np.rot90(P_BEADS)
		plt.scatter(plotxy[0], plotxy[1], s=bsize)
		if len(P_GLOB) != 0:
			for g in xrange(len(P_GLOB)):
				plt.scatter(P_GLOB[g][0], P_GLOB[g][1], s=P_GLOB[g][2] * 50, c='red')
		col += 1
	plt.axis([0,L,0,L])
	plt.savefig('{}/Figures/{}.png'.format(PATH, origres * nres + 1 + n))
	plt.close()

	#if TYPE_B == 0: rattle(Vm1, V_BEADS, R_BONDS)

	"""
	print "POSITIONS"
	print P_BEADS
	print "\n"

	print "VELOCITIES"
	print V_BEADS
	print "\n"

	print "BOND LENGTHS"
	print R_BONDS
	print "\n"

	print "FORCES"
	print F_BEADS
	print "\n"
	"""
	print "TEMP"
	print tempa, P_GLOB, len(P_GLOB)
	print "\n"

	if np.mod(n, nres) == 0: 
		save_restart(P_BEADS, V_BEADS, P_GLOB, Nch, Nbe, T, L, nsteps, res, PATH)
		res += 1

	if TYPE_G == 1:

		for i in xrange(len(P_GLOB)):
			if P_GLOB[i][2] <= maxglobs: P_GLOB[i][2] += cglob

		if np.random.random() <= Pglob and len(P_GLOB) < maxglobn: add_globule(P_GLOB, L, sig2)
	"""

fig = plt.figure()
ax = fig.gca(projection='3d')
for i in xrange(n_fibres):
	C1temp = np.rot90(C_x[i])
	ax.plot(C1temp[0], C1temp[1], C1temp[2])
	ax.scatter(C1temp[0], C1temp[1], C1temp[2])
plt.show()
"""
print "AVERAGE TEMPERATURE = {}".format(Tav)




