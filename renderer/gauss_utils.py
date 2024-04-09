from plyfile import PlyData
import numpy as np


class Gaussians:
    def __init__(self, positions, scales, rotations, shs, opacities):
        self.position = positions
        self.scale = scales
        self.rotation = rotations
        self.sh = shs
        self.opacity = opacities

    
    @staticmethod
    def load_gaussians(file_path):

        with open(file_path, 'rb') as f:
            plydata = PlyData.read(f)

        # for element in plydata.elements[0].data:
        #     print(element)
        plydata = plydata.elements[0]

        # properties of each vertex in the file are:
        #   - x, y, z; position
        #   - nx, ny, nz; normals, all 0 so ignore
        #   - opacity
        #   - scale_0, _1, _2; scaling factors
        #   - rot_0, _1, _2, _3; rotation factors
        #   - f_dc_0, _1, _2; principle SHs (just plain rgb colour?)
        #   - f_rest_1, ..., _44; rest of the harmonics
            
        # permanently ignore normals, don't need them
        
        positions = np.stack((np.asarray(plydata['x']), np.asarray(plydata['y']), np.asarray(plydata['z'])), axis=1)

        scales = np.column_stack((np.asarray(plydata['scale_0']), np.asarray(plydata['scale_1']), np.asarray(plydata['scale_2'])))
        rotations = np.column_stack((np.asarray(plydata['rot_0']), np.asarray(plydata['rot_1']), np.asarray(plydata['rot_2']),np.asarray(plydata['rot_3'])))
        opacities = np.array(plydata['opacity'])

        # start with basic RGB components only (only 3 values)
        shs = np.column_stack([*[plydata[f'f_dc_{n}'] for n in range(3)]])#, *[plydata[f'f_rest_{m}'] for m in range(9)]])
        print(shs[0])

        # might need to exp scales, normalise rotations, sigmoid opacities
        # need to normalise the rots to get a valid unit quaternion
        positions = positions.astype(np.float32)
        scales = np.exp(scales)
        scales = scales.astype(np.float32)
        rotations = (rotations / np.linalg.norm(rotations, axis=-1, keepdims=True))
        rotations = rotations.astype(np.float32)
        opacities = (1 / (1 + np.exp(-opacities))) # sigmoid
        opacities = opacities.astype(np.float32) # sigmoid
        shs = shs.astype(np.float32)

        return Gaussians(positions, scales, rotations, shs, opacities)


if __name__ == "__main__":
    g = Gaussians.load_gaussians("C:/Users/kirst/Downloads/point_cloud(7).ply")
    print(max(-g.position[:,0]))
