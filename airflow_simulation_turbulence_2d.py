import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

# Read the STL file
mesh = pv.read('building.stl')

# Get the bounds of the mesh to define the grid
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

# Grid resolution
nx, ny, nz = 120, 120, 3  # Increased nz to 3

# Create a uniform grid around the building
x = np.linspace(xmin - 20, xmax + 40, nx)
y = np.linspace(ymin - 20, ymax + 20, ny)
z = np.linspace(zmin + 2, zmin + 2 + 0.2, nz)  # Slight variation in z

# Create the meshgrid
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# Flatten the arrays to create point coordinates
points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

# First Pass: Generate initial airflow and streamlines
# ---------------------------------------------------

# Create a PyVista StructuredGrid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [nx, ny, nz]

# Compute the implicit distance from each point in the grid to the mesh
grid_with_distance = grid.compute_implicit_distance(mesh)
signed_distances = grid_with_distance['implicit_distance']

# Initialize uniform wind vectors pointing in the x-direction
vectors = np.zeros_like(points)
vectors[:, 0] = 1  # Wind in x-direction
vectors[:, 2] = 0  # Zero vertical component for 2D flow

# Compute the gradient of the distance field
distance_field = signed_distances.reshape((nx, ny, nz))
grad_x, grad_y, grad_z = np.gradient(
    distance_field,
    x[1] - x[0],
    y[1] - y[0],
    z[1] - z[0],
    edge_order=2
)

# Flatten the gradients
grad_x = grad_x.ravel()
grad_y = grad_y.ravel()
grad_z = grad_z.ravel()

# Combine gradients
gradients = np.column_stack((grad_x, grad_y, grad_z))
gradient_norms = np.linalg.norm(gradients, axis=1) + 1e-6  # Avoid division by zero
normals = gradients / gradient_norms[:, np.newaxis]

# Project the uniform wind onto the plane perpendicular to the normals to get tangential vectors
uniform_wind = vectors.copy()
dot_products = np.einsum('ij,ij->i', uniform_wind, normals)
tangential_wind = uniform_wind - dot_products[:, np.newaxis] * normals

# Normalize tangential wind vectors
tangential_norms = np.linalg.norm(tangential_wind, axis=1) + 1e-6
tangential_wind /= tangential_norms[:, np.newaxis]

# Define a smooth weighting function based on distance
threshold = 10.0  # Distance within which wind vectors are adjusted
distances = np.abs(signed_distances)
weights = np.clip((threshold - distances) / threshold, 0, 1)
weights = weights ** 2  # Squared for smoother transition

# Blend uniform wind with tangential wind using weights
vectors = (1 - weights[:, np.newaxis]) * uniform_wind + weights[:, np.newaxis] * tangential_wind

# Normalize the resulting vectors
vector_norms = np.linalg.norm(vectors, axis=1) + 1e-6
vectors /= vector_norms[:, np.newaxis]

# Set wind vectors inside the building to zero
inside_building = signed_distances < 0
vectors[inside_building] = 0

# Since we're in 2D, set the z-component of the vectors to zero
vectors[:, 2] = 0

# Assign the wind vectors to the grid
grid['vectors'] = vectors

# Adjust seed points to be within the grid bounds
seed_x = np.full(num_seeds, x.min() + 5)
seed_y = np.linspace(y.min() + 5, y.max() - 5, num_seeds)
seed_z = np.full(num_seeds, z[0])  # Use the first z-value from the grid
seed_points = np.column_stack((seed_x, seed_y, seed_z))
seeds = pv.PolyData(seed_points)

# Interpolate vectors at seed points to check if they are non-zero
seed_vectors = grid.interpolate(seed_points)['vectors']
if np.all(np.linalg.norm(seed_vectors, axis=1) == 0):
    print("All seed vectors are zero at the seed points. Streamlines cannot be generated.")
else:
    print("Seed vectors are non-zero at the seed points.")

# Generate streamlines from the seed points
initial_streamlines = grid.streamlines_from_source(
    seeds,
    vectors='vectors',
    integration_direction='forward',
    max_steps=5000,
    max_error=1e-8,
    terminal_speed=1e-12,
    integrator_type=45,  # Use default Runge-Kutta 4/5 integrator
    initial_step_length=0.5,
    step_unit='cl',
)

# Check if any streamlines were generated
if initial_streamlines.n_points == 0:
    print("No streamlines were generated in the first pass.")
    # Handle the case when no streamlines are generated
    # Exit the script or adjust parameters
    exit()
else:
    print(f"Number of points in initial streamlines: {initial_streamlines.n_points}")

# (Continue with the rest of the script as before)
