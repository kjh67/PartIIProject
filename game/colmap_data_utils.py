import numpy as np
import struct
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def read_next_bytes(file_id, num_bytes, format_sequence, endian_character="<"):
    data = file_id.read(num_bytes)
    return struct.unpack(endian_character + format_sequence, data)


def read_image_positions(path):
    """Returns positions, in 3D world coordinates, at which each image in the binary 
    file identified by path argument was taken, as approximated by COLMAP SFM."""

    quaternions = []
    translations = []

    with open(path, 'rb') as f:
        number_of_images = read_next_bytes(f, num_bytes=8, format_sequence="Q")[0]
        for _ in range(number_of_images):
            image_properties = read_next_bytes(f, num_bytes=64, format_sequence="idddddddi")
            quaternions.append(np.array(image_properties[1:5]))
            translations.append(image_properties[5:8])
            current_char = read_next_bytes(f, 1, "c")[0]
            while current_char != b"\x00":
                current_char = read_next_bytes(f, 1, "c")[0]
            num_points = read_next_bytes(f, num_bytes=8, format_sequence="Q")[0]
            f.seek(24*num_points, 1) # offset, seek mode 1 (from current position)

    # convert quaternions to rotation matrices, make sure translations are in a numpy array
    quaternions = np.array(quaternions)
    rotations = Rotation.from_quat(quaternions).as_matrix()
    translations = np.array(translations)

    # multiple each rotation matrix with corresponding translation vector
    points = np.einsum("ijk,ik->ij", rotations, translations)

    return points


def fit_plane_to_points(points):
    """Returns the coefficients of the equation of the plane best fitting the given points;
    plane equation is of the form (ax + by + cz + d = 0)."""

    def sum_squared_distances(coeffs):
        # for each point; do dot product with first three of coefs, add d
        # then divide everything by sum of coefs (without d) squared
        # then sum it all up
        return (np.einsum("ij,j->i", points, coeffs[:3]) + coeffs[3]) / np.einsum("i,i",coeffs[:3],coeffs[:3])

    # least squares optimisation on sum_squared_distances
    initial = np.array([0,0,1,0]) # xy plane initial guess
    coefficients = least_squares(sum_squared_distances, initial)

    # scale so that the first 3 coefficients are a unit normal
    return coefficients.x / np.linalg.norm(coefficients.x[:3])


def generate_model_matrix(path_to_file):
    """Takes coefficients of the equation of the movement plane (ax + by + cz + d = 0),
    and returns the model matrix aligning this plane with the xy plane."""

    points = read_image_positions(path_to_file)
    coefficients = fit_plane_to_points(points)

    normal = coefficients[:3] # already a unit vector
    print(f'Normal:{normal}')

    z_translation = -coefficients[3]/coefficients[2]

    # translation to origin
    translation = np.identity(4)
    translation[2][3] = -z_translation

    # AIM; align normals with the positive y direction
    # first rotate to zero the x coord, then to zero the z coord

    # Rotate about x to get a z coord of 0

    # find angle between the normal's projection into the zy plane and the z axis
    # so, set x coord to 0 and normalise
    zy_projection = np.array([0, normal[1], normal[2]])
    print(zy_projection)
    zy_projection = zy_projection / np.linalg.norm(zy_projection)
    print(zy_projection)

    # dot product with y axis will just be the y component
    theta = -np.arccos(zy_projection[1])

    # rotate by theta about the x axis
    rotation_x = np.array([[1,0,0],
                           [0,np.cos(theta), -np.sin(theta)],
                           [0, np.sin(theta), np.cos(theta)]])


    # apply rotation to the normal
    normal = np.matmul(rotation_x, normal)
    print("Z coord ")
    print(normal)


    # Rotate about z to get an x coord of 0
    xy_projection = np.array([normal[0], normal[1], 0])
    xy_projection = xy_projection / np.linalg.norm(xy_projection)
    print(xy_projection)

    # dot product with the y axis will be just the y component
    theta = np.arccos(xy_projection[1])

    # rotate by theta about the z axis
    rotation_z = np.array([[np.cos(theta), np.sin(theta), 0],
                           [-np.sin(theta), np.cos(theta), 0],
                           [0,0,1]])
    
    # apply rotation to the normal
    normal = np.matmul(rotation_z, normal)
    print('Only y coord should be 1')
    print(normal)

    all_rotation = np.matmul(rotation_z, rotation_x)

    print('checking normal')
    print(np.matmul(all_rotation, coefficients[:3]))


    print(np.linalg.det(all_rotation))

    # final corrective rotation; 90 degrees about x axis

    correction = np.array([[1,0,0],
                           [0,np.cos(-np.pi/2), np.sin(-np.pi/2)],
                           [0,-np.sin(-np.pi/2), np.cos(-np.pi/2)]])
    
    all_rotation = np.matmul(correction, all_rotation)

    # invert the rotations
    all_rotation = np.linalg.inv(all_rotation)

    # convert matrix to homogeneous coordinates
    model_matrix =  np.array([[*all_rotation[0], 0],
                     [*all_rotation[1], 0],
                     [*all_rotation[2], 0],
                     [0,0,0,1]])

    return model_matrix


if __name__ == "__main__":
    generate_model_matrix("./images.bin")

