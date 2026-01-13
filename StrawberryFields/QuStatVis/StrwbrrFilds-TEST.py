import strawberryfields as sf
from strawberryfields.ops import Ket, BSgate, Interferometer, CKgate, Sgate, S2gate, CZgate, Vgate, Fouriergate
#import strawberryfields.circuitdrawer as cdraw
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cutoff_dim = 3  # (1+ total number of photons)pip 
paths = 4
modes = 2 * paths

initial_state = np.zeros([cutoff_dim] * modes, dtype=np.complex_)
# The ket below corresponds to a single horizontal photon in each of the modes
initial_state[1, 0, 1, 0, 1, 0, 1, 0] = 1
# Permutation matrix
X = np.array([[0, 1], [1, 0]])
Y = np.array([[1, 0], [0, 1]])
Z = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])
U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0 , 1/np.sqrt(2), 1/np.sqrt(2)], [0, 0 , 1/np.sqrt(2), -1/np.sqrt(2)]])

# Here is the main program
# We create the input state and then send it through a network of beamsplitters and swaps.
prog = sf.Program(8)
with prog.context as q:
    Ket(initial_state) | q  # Initial state preparation
    for i in range(paths):
        BSgate() | (q[2 * i], q[2 * i + 1])
        S2gate(0.52, np.pi/2) | (q[2 * i], q[2 * i + 1])  # First layer of beamsplitters
    Interferometer(X) | (q[1], q[3])
    Interferometer(X) | (q[5], q[7])
    BSgate() | (q[2], q[3])
    Vgate(4.13) | (q[0])
    for i in range(modes):
       Vgate(1.67*(i+1)) | (q[i])
    BSgate() | (q[4], q[5])
    Interferometer(Z, 'triangular') | (q[3], q[5])
    BSgate().H | (q[2], q[3])
    BSgate().H | (q[4], q[5])
    BSgate(np.pi/6).H | (q[0], q[6])
    Interferometer(Y, 'triangular') | (q[3], q[5])
    Interferometer(Y, 'triangular') | (q[7], q[1])
    CKgate(1.79*np.pi) | (q[0], q[7])
    Interferometer(Z) | (q[6], q[7])
    CKgate(4.13*np.pi).H | (q[0], q[1])
    Interferometer(U)| (q[0], q[1], q[6], q[7])
    Interferometer(U, 'triangular') | (q[2], q[3], q[4], q[5])
    for i in range(modes):
       Fouriergate() | (q[i])

# We run the simulation
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
result = eng.run(prog)
state = result.state
ket = state.ket()

# Check the normalization of the ket.
# This does give the exact answer because of the cutoff we chose.
print("The norm of the ket is ", np.linalg.norm(ket))

# Postselection patterns:
patterns = [
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
]

sub_ket1 = np.round(ket[:, :, :, :, :, :, :, :], 5)  # postselect on correct pattern
p1 = np.round(np.linalg.norm(sub_ket1) ** 2, 5)  # Check the probability of this event
print("The probability is ", p1)
print("The expected probability is ", 1 / 32)

# These are the only nonzero components
ind1 = np.array(np.nonzero(np.real_if_close(sub_ket1))).T
print("The indices of the nonzero components are \n ", ind1)

# And these are their coefficients
#print("The nonzero components have values ", [sub_ket1[tuple(ind)] for ind in ind1])

# Transpose the ket
ket_t = ket.transpose(2, 3, 4, 5, 0, 1, 6, 7)

# sub_kets = [np.round(ket_t[tuple(ind)], 15) for ind in patterns]
# ps = np.array(list(map(np.linalg.norm, sub_kets))) ** 2
# indices = np.array([np.array(np.nonzero(sub_ket)).T for sub_ket in sub_kets])
# print("The indices of the nonzero components for the six different postselections are \n",indices,)

fig = plt.figure()
X = np.linspace(-10, 10, 400)
P = np.linspace(-10, 10, 400)
Z = state.wigner(0, X, P)
X, P = np.meshgrid(X, P)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, P, Z, cmap="PRGn", lw=0.5, rstride=1, cstride=1)
fig.set_size_inches(5.2, 5.2)
#ax.set_axis_off()
plt.show()

'''The successful postselection events occur with the same probability
print("The success probabilities for each pattern are \n", ps)
cdraw.Circuit.parse_op(modes, prog)
cdraw.Circuit.dump_to_document()'''
