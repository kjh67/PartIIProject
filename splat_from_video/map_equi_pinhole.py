import numpy as np
import cv2


def calculate_rot_matrix(yaw, pitch):

    # rotate through y
    pitch_mat = np.array([[np.cos(pitch), -np.sin(pitch), 0],
                        [np.sin(pitch), np.cos(pitch),  0],
                        [0,           0,            1]])
    
    # rotate through z
    yaw_mat = np.array([[np.cos(yaw),  0, np.sin(yaw)],
                          [0,              1, 0            ],
                          [-np.sin(yaw), 0, np.cos(yaw)]])
    
    return np.matmul(yaw_mat, pitch_mat)


def map_equi_pinhole(FOV, eq_width, eq_height, target_width, target_height, yaw, pitch):
    theta = np.radians(FOV)/2
    
    # width and height of projection plane
    pp_width = 2*np.tan(theta)
    pp_height = 2*np.tan(theta * target_height/target_width)

    # WIDTH = COLUMNS, HEIGHT = ROWS

    # create numpy arrays of x,y,z coords of projected points
    xs = np.ones((target_height, target_width))
    ys = np.tile(np.linspace(0, pp_height, target_height), (target_width,1)).transpose() - 0.5*pp_height
    zs = np.tile(np.linspace(0, pp_width, target_width), (target_height,1)) - 0.5*pp_width

    ks = np.sqrt(xs**2 + ys**2 + zs**2)

    # also reshape ready for rotation
    xs = np.reshape(xs/ks, (target_height*target_width))
    ys = np.reshape(ys/ks, (target_height*target_width))
    zs = -np.reshape(zs/ks, (target_height*target_width))

    uc_points = np.array([*zip(xs, ys, zs)])

    # put rotation stuff here
    # reshape before applying the rotation (need second shape to be 3)
    rot = calculate_rot_matrix(np.radians(yaw), np.radians(pitch))
    uc_points = np.dot(uc_points, rot.T)

    # get the xs, ys, xs back out
    xs = uc_points[:, 0]
    ys = uc_points[:, 1]
    zs = uc_points[:, 2]

    # angles from the origin in radians
    phis = np.arcsin(ys) + np.pi/2
    thetas = 3*np.pi - np.arctan2(zs, xs)

    # scale the phis and thetas to the equi image, in the correct range
    eq_y_mapping = np.reshape(((phis*eq_height)/np.pi), (target_height, target_width))
    eq_x_mapping = np.reshape((thetas*eq_width)/(2*np.pi), (target_height, target_width))

    return (eq_x_mapping, eq_y_mapping)

if __name__ == "__main__":
    mappings = map_equi_pinhole(100, 5760, 2880, 1600,1200,0,45)
    print('generated mappings')
    image = cv2.imread('./frames/frame0.jpg')
    output = cv2.remap(image, mappings[0].astype(np.float32), mappings[1].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    cv2.imwrite('test.jpg', output)