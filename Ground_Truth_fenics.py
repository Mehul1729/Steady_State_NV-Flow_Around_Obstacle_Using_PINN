import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from mshr import *

# ---------------------------------------------------
# 1. PARAMETERS & GEOMETRY
# ---------------------------------------------------
Re = 4.0
mu_val = 1.0 / Re
rho_val = 1.0

# Define the Domain (Rectangle [-8, 12] x [-8, 8])
domain = Rectangle(Point(-8.0, -8.0), Point(12.0, 8.0))

# Define the Obstacle (Ellipse with a=1.5, b=1.0 at origin)
# Since n=2.0 in the PINN, this is a standard ellipse
obstacle = Ellipse(Point(0.0, 0.0), 1.5, 1.0)

# Subtract obstacle from domain to get the fluid region
fluid_domain = domain - obstacle

# Generate a fine mesh (120 resolution creates a very dense, accurate grid)
print("Generating mesh...")
mesh = generate_mesh(fluid_domain, 120)
print(f"Mesh generated with {mesh.num_cells()} cells.")

# ---------------------------------------------------
# 2. FUNCTION SPACES (Taylor-Hood P2-P1 Elements)
# ---------------------------------------------------
# V is for Velocity (Vector, degree 2), Q is for Pressure (Scalar, degree 1)
V = VectorElement("CG", mesh.ufl_cell(), 2)
Q = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))

# ---------------------------------------------------
# 3. BOUNDARY CONDITIONS
# ---------------------------------------------------
# Define boundary identification functions
def inlet(x, on_boundary):
    return on_boundary and near(x[0], -8.0)

def outlet(x, on_boundary):
    return on_boundary and near(x[0], 12.0)

def walls_and_obstacle(x, on_boundary):
    # Anything on the boundary that isn't the inlet or outlet must be a wall or the obstacle
    return on_boundary and not (near(x[0], -8.0) or near(x[0], 12.0))

# Parabolic inlet: u = 1.0 * (1 - (y/8)^2)
inlet_velocity = Expression(('1.0 * (1.0 - pow(x[1]/8.0, 2))', '0.0'), degree=2)

bc_in = DirichletBC(W.sub(0), inlet_velocity, inlet)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls_and_obstacle)
bc_out = DirichletBC(W.sub(1), Constant(0.0), outlet) # P = 0 at outlet

bcs = [bc_in, bc_walls, bc_out]

# ---------------------------------------------------
# 4. VARIATIONAL FORMULATION
# ---------------------------------------------------
w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)

mu = Constant(mu_val)
rho = Constant(rho_val)

# Steady-state Navier-Stokes weak form
F = (rho * dot(dot(u, nabla_grad(u)), v) * dx
     + mu * inner(grad(u), grad(v)) * dx
     - p * div(v) * dx
     + div(u) * q * dx)

# ---------------------------------------------------
# 5. SOLVER
# ---------------------------------------------------
print("Solving Navier-Stokes equations...")
solve(F == 0, w, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6, "maximum_iterations": 50}})
print("Solve complete.")

# Extract velocity and pressure functions
u_sol, p_sol = w.split()

# Save FEniCS data to Paraview format (optional, for 3D viewing)
File("results/velocity.pvd") << u_sol
File("results/pressure.pvd") << p_sol

# ---------------------------------------------------
# 6. EXPORTING TO NUMPY GRID (For PINN Comparison)
# ---------------------------------------------------
print("Interpolating FEniCS solution onto regular grid...")

# Create the exact same grid used in your PINN plotting script
# We'll stick to the crop region [-8, 12] directly to save memory
x_grid = np.linspace(-8, 12, 400)
y_grid = np.linspace(-8, 8, 320)
X, Y = np.meshgrid(x_grid, y_grid)

U_fenics = np.zeros_like(X)
V_fenics = np.zeros_like(X)
P_fenics = np.zeros_like(X)

# Sample the FEniCS solution at every grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_val, y_val = X[i, j], Y[i, j]
        
        # Check if point is inside the ellipse obstacle
        if (x_val/1.5)**2 + (y_val/1.0)**2 <= 1.0:
            U_fenics[i, j] = np.nan
            V_fenics[i, j] = np.nan
            P_fenics[i, j] = np.nan
        else:
            try:
                # Evaluate FEniCS functions at the exact coordinate
                pt = Point(x_val, y_val)
                vel = u_sol(pt)
                press = p_sol(pt)
                
                U_fenics[i, j] = vel[0]
                V_fenics[i, j] = vel[1]
                P_fenics[i, j] = press
            except:
                # Handle points exactly on the mesh boundary edge
                U_fenics[i, j] = np.nan
                V_fenics[i, j] = np.nan
                P_fenics[i, j] = np.nan

# Save the arrays
np.save("U_fenics.npy", U_fenics)
np.save("V_fenics.npy", V_fenics)
np.save("P_fenics.npy", P_fenics)
print("Data saved successfully as .npy files.")

# ---------------------------------------------------
# 7. QUICK VISUALIZATION
# ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

c1 = axes[0].contourf(X, Y, U_fenics, levels=50, cmap='jet')
axes[0].set_title("FEniCS Ground Truth: U-Velocity")
fig.colorbar(c1, ax=axes[0])

c2 = axes[1].contourf(X, Y, P_fenics, levels=50, cmap='coolwarm')
axes[1].set_title("FEniCS Ground Truth: Pressure")
fig.colorbar(c2, ax=axes[1])

plt.tight_layout()
plt.savefig("fenics_ground_truth.png")
print("Saved ground truth plot as fenics_ground_truth.png")