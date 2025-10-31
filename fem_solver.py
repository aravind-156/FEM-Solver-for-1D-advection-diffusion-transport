import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from matplotlib.animation import FuncAnimation

# Parameters
L = 1                       # Length of domain (m)
alpha = 1e-3                # Effective diffusivity (m^2/s)
U = 0.1                     # Fluid velocity (m/s)

# Discretization
Nx = 101                      # Nodes
Ne = Nx - 1                   # Elements 
dx = L/Ne                     # Length of element

dt = 0.01                        # Time step (s)
t_final = 16                 # Total simulation time (s)

ani_interval = 10             # Data to be collected for animation after every 10 time steps   


# MATRIX FUNCTIONS FOR EACH ELEMENT

# Time derivative matrix 
def time_deri_matrix(dx):
    return (dx/6) * np.array([[2,1], [1,2]])

# Diffusion matrix 
def diffusion_matrix(alpha, dx):
    return (alpha/dx) * np.array([[1,-1], [-1,1]])

# Advection matrix
def advection_matrix(U, dx):
    return (U/2) * np.array([[-1,1], [-1,1]])


# Creating global matrices (each of size 101x101)
TD_global = np.zeros((Nx, Nx))                     # Global matrix for time derivative term
D_global = np.zeros((Nx, Nx))                      # Global matrix for diffusion term
A_global = np.zeros((Nx, Nx))                      # Global matrix for advection term

# Assembly of global matrices 
for i in range(Ne):
    TD_i = time_deri_matrix(dx)                    # 2x2 elemental matrices
    D_i = diffusion_matrix(alpha, dx)
    A_i = advection_matrix(U, dx)

    TD_global[i, i] += TD_i[0,0]                   # Top left
    TD_global[i, i+1] += TD_i[0,1]                 # Top right
    TD_global[i+1, i] += TD_i[1,0]                 # Bottom left
    TD_global[i+1, i+1] += TD_i[1,1]               # Bottom right

    D_global[i, i] += D_i[0,0]                     # Top left
    D_global[i, i+1] += D_i[0,1]                   # Top right
    D_global[i+1, i] += D_i[1,0]                   # Bottom left
    D_global[i+1, i+1] += D_i[1,1]                 # Bottom right

    A_global[i, i] += A_i[0,0]                     # Top left
    A_global[i, i+1] += A_i[0,1]                   # Top right
    A_global[i+1, i] += A_i[1,0]                   # Bottom left
    A_global[i+1, i+1] += A_i[1,1]                 # Bottom right


# Initial and boundary conditions
phi = np.zeros(Nx)                    # Initially 0 concentration everywhere
phi[0] = 1                            # Concentration at left end

phi_initial = phi.copy()
phi_data = [phi_initial.copy()]       # This list to be used for animation


# A matrix 
A = (TD_global/dt) + D_global + A_global               # This matrix is constant

A[0, :] = 0.0     # Set the entire first row to zeros
A[0, 0] = 1.0     # Set the diagonal element to 1


num_steps = int(t_final/dt)
print(f"Number of time steps in simulation: {num_steps}")

t_inj = 1
t_graph1 = 1
t_graph2 = 3
t_graph3 = 6
t_graph4 = 10

for i in range(num_steps):
    t = i*dt
    b = TD_global.dot(phi) / dt
    if t <= t_inj:
        b[0] = 1
    else:
        b[0] = 0

    phi_new = solve(A, b)
    phi = phi_new.copy()

    if int(dt*i) == t_graph1:
        phi_graph1 = phi.copy()
    elif int(dt*i) == t_graph2:
        phi_graph2 = phi.copy()
    elif int(dt*i) == t_graph3:
        phi_graph3 = phi.copy()
    elif int(dt*i) == t_graph4:
        phi_graph4 = phi.copy()

    if i % ani_interval == 0:
        phi_data.append(phi.copy())

print("Simulation complete")


# Plotting
x = np.linspace(0,L,Nx)
plt.figure(figsize = (10,6))
plt.plot(x, phi_graph1 , 'b-', label = f"Concentration at t = {t_graph1}s")
plt.plot(x, phi_graph2 , 'g-', label = f"Concentration at t = {t_graph2}s")
plt.plot(x, phi_graph3 , 'm-', label = f"Concentration at t = {t_graph3}s")
plt.plot(x, phi_graph4 , 'y-', label = f"Concentration at t = {t_graph4}s")
plt.plot(x,phi_initial, 'r--', label = "Initial concentration (t = 0s)")

plt.xlabel("Position (m)")
plt.ylabel("Ink concentration")
plt.title("1D Advection-Diffusion FEM solver")
plt.legend()
plt.grid(True)
plt.show()


# ANIMATION

time_points = []
for i in range(len(phi_data)):
    current_time = i * ani_interval * dt
    time_points.append(current_time)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
ax.set_ylim(-0.1, 1.2)  # Set fixed Y-axis limits
ax.set_xlabel("Position (m)")
ax.set_ylabel("Ink concentration")
ax.set_title("1D Ink Puff Simulation")
ax.grid(True)

line, = ax.plot(x, phi_data[0], 'b-', lw=2, label="Ink Concentration")
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()

# Update function
def update(frame):
    line.set_ydata(phi_data[frame])
    time_text.set_text(f"Time = {time_points[frame]:.2f} s")
    return line, time_text

ani = FuncAnimation(fig, update, frames=len(phi_data),interval=50, blit=True)
ani.save('Ink_Puff_Animation.mp4', writer='ffmpeg', fps=20)
plt.show()