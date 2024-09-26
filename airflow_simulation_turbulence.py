import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

pv.global_theme.allow_empty_mesh = True

# Read the STL file
mesh = pv.read('building.stl')

# Get the bounds of the mesh to define the grid
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

# Grid resolution
nx, ny, nz = 120, 120, 60  # Adjust as needed

# Create a uniform grid around the building
x = np.linspace(xmin - 20, xmax + 40, nx)
y = np.linspace(ymin - 20, ymax + 20, ny)
z = np.linspace(zmin - 10, zmax + 20, nz)
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

# Compute the gradient of the distance field
distance_field = signed_distances.reshape((nx, ny, nz))
grad_x, grad_y, grad_z = np.gradient(
    distance_field,
    x[1] - x[0],
    y[1] - y[0],
    z[1] - z[0],
    edge_order=2
)
gradients = np.column_stack((
    grad_x.ravel(),
    grad_y.ravel(),
    grad_z.ravel()
))
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

# Assign the wind vectors to the grid
grid['vectors'] = vectors

# Define seed points for streamlines at 2m above the ground
num_seeds = 50
seed_x = np.full(num_seeds, xmin - 15)  # Start before the building
seed_y = np.linspace(ymin - 20, ymax + 20, num_seeds)
seed_z = np.full(num_seeds, zmin + 2)  # 2m above ground level
seed_points = np.column_stack((seed_x, seed_y, seed_z))
seeds = pv.PolyData(seed_points)

# Generate streamlines from the seed points
initial_streamlines = grid.streamlines_from_source(
    seeds,
    vectors='vectors',
    integration_direction='forward',
    max_steps=5000,
    max_error=1e-6,
    terminal_speed=1e-10,
    integrator_type=45,
    initial_step_length=0.5,
    step_unit='cl',
)

# Extract all initial streamline points
initial_streamline_points = initial_streamlines.points

# Create a KDTree of initial streamline points
initial_streamline_tree = cKDTree(initial_streamline_points)

# Candidate points: All points outside the building (signed_distances >= 0)
candidate_points_mask = signed_distances >= 0
candidate_points = points[candidate_points_mask]
candidate_indices = np.where(candidate_points_mask)[0]

# Find candidate points that are far from any streamline
distance_threshold = 2.75  # Adjust as needed
distances_to_streamlines, _ = initial_streamline_tree.query(candidate_points, distance_upper_bound=distance_threshold)
no_streamline_nearby = np.isinf(distances_to_streamlines)  # Points with no streamline within threshold

# Identify the candidate points that lack nearby streamlines
sparse_points_first_pass = candidate_points[no_streamline_nearby]
sparse_indices_first_pass = candidate_indices[no_streamline_nearby]

# Filter sparse points at z = 2 meters (First Pass)
z_target = zmin + 2  # Assuming zmin is ground level
z_tolerance = 1.0    # Tolerance for floating-point comparison
z_coords_first_pass = sparse_points_first_pass[:, 2]
mask_z_first_pass = np.abs(z_coords_first_pass - z_target) <= z_tolerance
sparse_points_z2_first_pass = sparse_points_first_pass[mask_z_first_pass]

# Create a PolyData object for the sparse points at z = 2 meters (First Pass)
sparse_points_polydata_first_pass = pv.PolyData(sparse_points_z2_first_pass)

# Second Pass: Identify sparse regions and adjust vector field
# ------------------------------------------------------------

# If there are sparse points, adjust the vector field to attract streamlines
if len(sparse_points_first_pass) > 0:
    # Compute vectors towards the nearest streamline point
    distances, indices = initial_streamline_tree.query(points[sparse_indices_first_pass])
    nearest_streamline_points = initial_streamline_points[indices]
    direction_to_streamline = nearest_streamline_points - points[sparse_indices_first_pass]
    direction_norms = np.linalg.norm(direction_to_streamline, axis=1) + 1e-6
    direction_to_streamline /= direction_norms[:, np.newaxis]

    # Define weights for attraction based on distance to the nearest streamline
    attraction_strength = 10.5  # Adjust as needed
    vectors[sparse_indices_first_pass] = (1 - attraction_strength) * vectors[sparse_indices_first_pass] + attraction_strength * direction_to_streamline

    # Introduce angular momentum (rotational component)
    angular_strength = 0.2  # Adjust as needed
    for idx in sparse_indices_first_pass:
        # Create a rotational vector perpendicular to the wind direction
        wind_dir = vectors[idx]
        perp_vector = np.cross(wind_dir, [0, 0, 1])
        perp_norm = np.linalg.norm(perp_vector) + 1e-6
        perp_vector /= perp_norm
        vectors[idx] += angular_strength * perp_vector

    # Normalize the vectors at sparse points
    vector_norms = np.linalg.norm(vectors[sparse_indices_first_pass], axis=1) + 1e-6
    vectors[sparse_indices_first_pass] /= vector_norms[:, np.newaxis]

# Update the grid with the modified vectors
grid['vectors'] = vectors

# Regenerate streamlines with the updated vector field
streamlines = grid.streamlines_from_source(
    seeds,
    vectors='vectors',
    integration_direction='forward',
    max_steps=5000,
    max_error=1e-6,
    terminal_speed=1e-10,
    integrator_type=45,
    initial_step_length=0.5,
    step_unit='cl',
)

# Extract all streamline points after second pass
streamline_points = streamlines.points

# Create a KDTree of streamline points after second pass
streamline_tree = cKDTree(streamline_points)

# Find candidate points that are far from any streamline after second pass
distances_to_streamlines_after, _ = streamline_tree.query(candidate_points, distance_upper_bound=distance_threshold)
no_streamline_nearby_after = np.isinf(distances_to_streamlines_after)  # Points with no streamline within threshold

# Identify sparse points after second pass
sparse_points_second_pass = candidate_points[no_streamline_nearby_after]
sparse_indices_second_pass = candidate_indices[no_streamline_nearby_after]

# Filter sparse points at z = 2 meters (Second Pass)
z_coords_second_pass = sparse_points_second_pass[:, 2]
mask_z_second_pass = np.abs(z_coords_second_pass - z_target) <= z_tolerance
sparse_points_z2_second_pass = sparse_points_second_pass[mask_z_second_pass]

# Create a PolyData object for the sparse points at z = 2 meters (Second Pass)
sparse_points_polydata_second_pass = pv.PolyData(sparse_points_z2_second_pass)

# Visualization
# -------------

# Create a plotter with 1 row and 2 columns
plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

# First Pass Visualization
# ------------------------
plotter.subplot(0, 0)
plotter.add_title("First Pass - Initial Streamlines")
plotter.add_mesh(mesh, color='lightgray', opacity=1.0)
plotter.add_mesh(initial_streamlines.tube(radius=0.05), color='blue')

# Add the sparse points at z = 2 meters to the plotter (First Pass)
plotter.add_mesh(
    sparse_points_polydata_first_pass,
    color='red',
    point_size=5.0,
    render_points_as_spheres=True,
    opacity=1.0
)

# Adjust the camera to focus on the data range
plotter.camera_position = 'xy'
plotter.camera.focal_point = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
plotter.camera.position = [plotter.camera.focal_point[0], plotter.camera.focal_point[1], zmax + 100]
plotter.camera_set = True  # Ensure the camera settings are applied
plotter.reset_camera()
plotter.camera.zoom(1.5)

# Show the bounds with the specified data range
plotter.show_bounds(
    grid='front',
    location='all',
    bounds=[xmin, xmax, ymin, ymax, zmin, zmax],
    xlabel='X', ylabel='Y', zlabel='Z',
    minor_ticks=True,
    all_edges=True
)

# Second Pass Visualization
# -------------------------
plotter.subplot(0, 1)
plotter.add_title("Second Pass - Adjusted Streamlines")
plotter.add_mesh(mesh, color='lightgray', opacity=1.0)
plotter.add_mesh(streamlines.tube(radius=0.05), color='blue')

# Add the sparse points at z = 2 meters to the plotter (Second Pass)
plotter.add_mesh(
    sparse_points_polydata_second_pass,
    color='red',
    point_size=5.0,
    render_points_as_spheres=True,
    opacity=1.0
)

# Adjust the camera to focus on the data range
plotter.camera_position = 'xy'
plotter.camera.focal_point = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
plotter.camera.position = [plotter.camera.focal_point[0], plotter.camera.focal_point[1], zmax + 100]
plotter.camera_set = True  # Ensure the camera settings are applied
plotter.reset_camera()
plotter.camera.zoom(1.5)

# Show the bounds with the specified data range
plotter.show_bounds(
    grid='front',
    location='all',
    bounds=[xmin, xmax, ymin, ymax, zmin, zmax],
    xlabel='X', ylabel='Y', zlabel='Z',
    minor_ticks=True,
    all_edges=True
)

# Display the plots
plotter.show()

# Save the visualization to disk
plotter.screenshot('airflow_simulation_comparison.png')