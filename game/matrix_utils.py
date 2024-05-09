import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from colmap.scripts.python.read_write_model import read_images_binary


def get_image_positions(path):
    image_information = read_images_binary(path)
    positions = np.array([[0,0,0]])
    for imageid in image_information.keys():
        tvec = image_information[imageid].tvec
        quats = image_information[imageid].qvec
        rotation = Rotation.from_quat([quats[1], quats[2], quats[3], quats[0]]).as_matrix()
        positions = np.append(positions, [rotation.transpose() @ tvec], axis=0)
    print(positions)
    return -np.array(positions)


def fit_plane_to_points(points):
    """Returns the coefficients of the equation of the plane best fitting the given points;
    plane equation is of the form (ax + by + cz + d = 0)."""

    def mean_squared_distances(coeffs):
        # Calculate the squared distance from each point to the plane defined by coeffs
        # Minimise the mean distance
        distances = np.square(np.einsum("ij,j->i", points, coeffs[:3]) + coeffs[3]) / np.einsum("i,i",coeffs[:3],coeffs[:3])
        return np.mean(distances)

    # Perform least squares optimisation to find plane coefficients
    initial = np.array([0,1,0,0]) # xy plane initial guess
    coefficients = least_squares(mean_squared_distances, initial, jac='3-point')

    # Scale coefficients to give a unit normal
    return coefficients.x / np.linalg.norm(coefficients.x[:3])


def generate_movement_axes_from_colmap(path_to_file):
    """Takes coefficients of the equation of the movement plane (ax + by + cz + d = 0),
    and returns 1) the coefficients for the cartesian equation of this plane, and 2) the 
    4D rotation matrix aligning its normal with the y axis"""

    points = get_image_positions(path_to_file)
    print("COLMAP points:")
    print(points)
    plane_coefficients = fit_plane_to_points(points)

    # correct for OpenGL coordinate system
    plane_coefficients[2] = -plane_coefficients[2]

    # 'Up' vector is set to be the plane's normal
    up_vector = np.array(plane_coefficients[:3]) # already a unit vector

    # forwards direction will be an 'arbitrary' direction on the plane, i.e. 
    forward_direction = np.array([plane_coefficients[3] / plane_coefficients[0],
                                    -plane_coefficients[3] / plane_coefficients[1],
                                    0])
    forward_vector = forward_direction / np.linalg.norm(forward_direction)
    side_vector = np.cross(forward_vector, up_vector)


    # Create rotation matrix to align the default viewing direction (z axis) with forward vector
    # First, zero the x coordinate, by rotating about the y axis
    xz_projection = np.array([forward_vector[0], 0, forward_vector[2]])
    xz_projection = xz_projection / np.linalg.norm(xz_projection)

    # Find the angle through which to rotate; dot product with the z axis
    y_theta = -np.arccos(xz_projection[2])
    y_rotation = np.array([[np.cos(y_theta), 0, np.sin(y_theta)],
                        [0, 1, 0],
                        [-np.sin(y_theta), 0, np.cos(y_theta)]])
    rotated = y_rotation @ forward_vector
    print(f"After rotating through y axis (x should be 0): {rotated}")

    # Next, zero the y coordinate; rotate about the x axis
    yz_projection = np.array([0, rotated[1], rotated[2]])
    yz_projection = yz_projection / np.linalg.norm(yz_projection)

    # Find the angle to rotate; dot product with the z axis
    x_theta = np.arccos(yz_projection[2])
    x_rotation = np.array([[1,0,0],
                        [0, np.cos(x_theta), -np.sin(x_theta)],
                        [0, np.sin(x_theta), np.cos(x_theta)]])
    
    rotated = x_rotation @ rotated
    print(f"After rotating through x axis (y should be 0): {rotated}")


    # Final adjustment; find angle about z axis to align side_vector with the x axis
    side_vector_adjusted = x_rotation @ y_rotation @ side_vector
    print(f"Side vector aligned to new forward vector: {side_vector_adjusted}")
    z_theta = -np.arccos(-side_vector_adjusted[0])
    z_rotation = np.array([[np.cos(z_theta), -np.sin(z_theta), 0],
                           [np.sin(z_theta), np.cos(z_theta), 0],
                           [0, 0, 1]])
    side_vector_adjusted = z_rotation @ side_vector_adjusted
    print(f"Side vector after adjustment: {side_vector_adjusted}")
    # z_rotation = np.identity(3)

    alignment_rotation = np.linalg.inv(z_rotation @ x_rotation @ y_rotation)
    up_vector = np.array([*up_vector, 1])
    forward_vector = np.array([*forward_vector, 1])
    side_vector = np.array([*side_vector, 1])

    return up_vector, forward_vector, side_vector, np.array([[*alignment_rotation[0], 0],
                                                             [*alignment_rotation[1], 0],
                                                             [*alignment_rotation[2], 0],
                                                             [0,0,0,1]])


def get_rotation_about_axis(theta, axis):
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    [x,y,z,_] = axis
    rotation = np.array([
        [t*x*x + c, t*x*y -s*z, t*x*z + s*y, 0],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x, 0],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c, 0],
        [0,0,0,1]
    ])
    return rotation
