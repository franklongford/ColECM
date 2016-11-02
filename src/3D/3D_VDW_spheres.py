import numpy as np
import scipy.integrate as spin
import scipy.optimize as spop
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import 3D_utilities as ut

nsteps = 1000
N = 25
L = 15

sig1 = 2.
bsize = sig1 * 300
ep1 = 3.0
dt = 0.0005

#BEADS = ([4.,5.], [7.,3.], [10.,13.])
BEADS_HIS = np.zeros((N,nsteps+1,3))
P_BEADS = np.zeros((N,3))

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

V_BEADS = (np.random.random((N,3)) - 0.5) * 2
F_BEADS = np.zeros((N,3))

"""
potential = map (lambda x: POT(x, sig1, ep1), np.linspace(0,10,50))
plt.plot(np.linspace(0,10,50), potential)
plt.axis([0,10,-ep1-1,3])
plt.show()

plotxy = np.rot90(P_BEADS)

fig = plt.figure(0)
ax = fig.gca(projection='3d')
ax.scatter(plotxy[0], plotxy[1], plotxy[2], c=range(N), s=bsize)
ax.set_xlim3d(0, L)
ax.set_ylim3d(0, L)
ax.set_zlim3d(0, L)
plt.savefig('Figures/Spheres/0.png')

F_BEADS = calc_forces(N, P_BEADS, sig1, ep1)
"""

print "\n"
print "POSITIONS"
print P_BEADS
print "\n"

print "VELOCITIES"
print V_BEADS
print "\n"

"""
print "FORCES"
print F_BEADS
print "\n"
"""

for n in xrange(nsteps):

	for i in xrange(N):
		
		vx = V_BEADS[i][0] + 0.5 * dt * F_BEADS[i][0]
		vy = V_BEADS[i][1] + 0.5 * dt * F_BEADS[i][1]
		vz = V_BEADS[i][2] + 0.5 * dt * F_BEADS[i][2]

		P_BEADS[i][0] += dt * vx
		P_BEADS[i][1] += dt * vy
		P_BEADS[i][2] += dt * vy

		for j in xrange(3):
			a = P_BEADS[i][j]
			if P_BEADS[i][j] > L: 
				P_BEADS[i][j] -= int(P_BEADS[i][j] / L) * L
				#print "{} {}".format(a, P_BEADS[i][j])
			elif P_BEADS[i][j] < 0: 
				P_BEADS[i][j] -= int(P_BEADS[i][j] / L - 1) * L
				#print "{} {}".format(a, P_BEADS[i][j])

			BEADS_HIS[i][n+1][j] = P_BEADS[i][j]

		F_BEADS = calc_forces(N, P_BEADS, sig1, ep1)

		V_BEADS[i][0] = vx + 0.5 * dt * F_BEADS[i][0]
		V_BEADS[i][1] = vy + 0.5 * dt * F_BEADS[i][1]
		V_BEADS[i][2] = vy + 0.5 * dt * F_BEADS[i][2]

	"""
	plotxy = np.rot90(P_BEADS)
	fig = plt.figure(n+1)
	ax = fig.gca(projection='3d')
	ax.scatter(plotxy[0], plotxy[1], plotxy[2], c=range(N), s=bsize)
	ax.set_xlim3d(0, L)
	ax.set_ylim3d(0, L)
	ax.set_zlim3d(0, L)
	plt.savefig('Figures/Spheres/{}.png'.format(n+1))
	"""

	print "POSITIONS"
	print P_BEADS
	print "\n"

	print "VELOCITIES"
	print V_BEADS
	print "\n"

	"""
	print "FORCES"
	print F_BEADS
	print "\n"
	"""

print len(BEADS_HIS[0][0])


# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
#ax.axis('off')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim((0, L))
ax.set_ylim((0, L))
ax.set_zlim((0, L))


# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        #line.set_data([], [])
        #line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % BEADS_HIS.shape[1]

    for line, pt, xi in zip(lines, pts, BEADS_HIS):
        x, y, z = xi[:i].T
        #line.set_data(x, y)
        #line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    #ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nsteps+1, interval=10, blit=False)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()

"""

fig = plt.figure()
ax = fig.gca(projection='3d')

for dat in BEADS_HIS:
	print dat[0,0:1]
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

ax.set_xlim3d([0.0, L])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, L])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, L])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, nsteps+1, fargs=(BEADS_HIS, lines),
                              interval=50, blit=False)

plt.show()



fig = plt.figure()
ax = fig.gca(projection='3d')
for i in xrange(n_fibres):
	C1temp = np.rot90(C_x[i])
	ax.plot(C1temp[0], C1temp[1], C1temp[2])
	ax.scatter(C1temp[0], C1temp[1], C1temp[2])
plt.show()
"""


