import pyvista as pv
import numpy as np

# Read the STL file
mesh = pv.read('building.stl')

# Get the bounds of the mesh to define the grid
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

# Increased grid resolution
nx, ny, nz = 120, 120, 60  # Increased resolution for finer results

# Create a uniform grid around the building
x = np.linspace(xmin - 20, xmax + 20, nx)
y = np.linspace(ymin - 20, ymax + 20, ny)
z = np.linspace(zmin - 10, zmax + 20, nz)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# Flatten the arrays to create point coordinates
points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

# Create a PyVista StructuredGrid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [nx, ny, nz]

# Compute the implicit distance from each point in the grid to the mesh
grid_with_distance = grid.compute_implicit_distance(mesh)
signed_distances = grid_with_distance['implicit_distance']

# Reshape distances to the grid shape
distance_field = signed_distances.reshape((nx, ny, nz))

# Compute the gradient of the distance field
grad_x, grad_y, grad_z = np.gradient(
    distance_field,
    x[1] - x[0],
    y[1] - y[0],
    z[1] - z[0],
    edge_order=2
)

# Flatten the gradients
gradients = np.column_stack((
    grad_x.ravel(),
    grad_y.ravel(),
    grad_z.ravel()
))

# Normalize the gradients to get normals
gradient_norms = np.linalg.norm(gradients, axis=1) + 1e-6  # Avoid division by zero
normals = gradients / gradient_norms[:, np.newaxis]

# Initialize uniform wind vectors pointing in the x-direction
uniform_wind = np.zeros_like(points)
uniform_wind[:, 0] = 1  # Wind in x-direction

# Project the uniform wind onto the plane perpendicular to the normals to get tangential vectors
dot_products = np.einsum('ij,ij->i', uniform_wind, normals)
tangential_wind = uniform_wind - dot_products[:, np.newaxis] * normals

# Normalize tangential wind vectors
tangential_norms = np.linalg.norm(tangential_wind, axis=1) + 1e-6
tangential_wind /= tangential_norms[:, np.newaxis]

# Define a smooth weighting function based on distance
threshold = 10.0  # Distance within which wind vectors are adjusted
distances = np.abs(signed_distances)
weights = np.clip((threshold - distances) / threshold, 0, 1)
weights = weights ** 3  # Squared for smoother transition

# Blend uniform wind with tangential wind using weights
vectors = (1 - weights[:, np.newaxis]) * uniform_wind + weights[:, np.newaxis] * tangential_wind

# Normalize the resulting vectors
vector_norms = np.linalg.norm(vectors, axis=1) + 1e-6
vectors /= vector_norms[:, np.newaxis]

# Set wind vectors inside the building to zero
inside_building = signed_distances < 0
vectors[inside_building] = 0

# Assign the wind vectors to the grid
grid['vectors'] = vectors

# Adjust seed points
num_seeds = 100  # Increased number of seed points for better coverage
seed_x = np.full(num_seeds, xmin - 15)  # Start before the building
seed_y = np.linspace(ymin - 20, ymax + 20, num_seeds)
seed_z = np.full(num_seeds, zmin + 2)  # 2m above ground level
seed_points = np.column_stack((seed_x, seed_y, seed_z))

# Create a PyVista PolyData object for seed points
seeds = pv.PolyData(seed_points)

# Generate streamlines from the seed points
streamlines = grid.streamlines_from_source(
    seeds,
    vectors='vectors',
    integration_direction='forward',
    max_steps=4000,
    max_error=1e-6,
    terminal_speed=1e-10,
    integrator_type=45,
    initial_step_length=0.5,
    step_unit='cl',
)

# Visualize the building and streamlines
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='lightgray', opacity=1.0)  # Building mesh
plotter.add_mesh(streamlines.tube(radius=0.1), color='blue')  # Streamlines as tubes
plotter.show_grid()
plotter.camera_position = 'xy'  # Top-down view
plotter.show()

# Save the visualization to disk
plotter.screenshot('airflow_simulation.png')
