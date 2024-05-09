import numpy as np

# OpenGL commands
from OpenGL.GL import glUseProgram, glDisable, glEnable, glBlendFunc,\
    glGenBuffers, glGetAttribLocation, glBindBuffer, glBufferData,\
    glVertexAttribPointer, glEnableVertexAttribArray, glGenVertexArrays,\
    glBindVertexArray, glVertexAttribDivisor, glGetUniformLocation,\
    glUniformMatrix4fv, glUniform2fv, glUniformMatrix3fv, glClearColor,\
    glClear, glDrawArraysInstanced
# OpenGL enums
from OpenGL.GL import GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA,\
    GL_ONE_MINUS_SRC_ALPHA, GL_ARRAY_BUFFER, GL_FLOAT,\
    GL_STATIC_DRAW, GL_COLOR_BUFFER_BIT, GL_TRIANGLE_STRIP,\
    GL_TRUE

from renderer.renderer_template import Renderer


class GaussianRenderer(Renderer):
    vshader = "./renderer/gauss_vert.glsl"
    fshader = "./renderer/gauss_frag.glsl"
    
    def __init__(self, filepath, screenwidth, screenheight, mv_matrix=np.identity(4), fovx=50, fovy=50, znear=0.2, zfar=200):
        tanfovx = np.tan(np.deg2rad(fovx/2))
        tanfovy = np.tan(np.deg2rad(fovy/2))
        super().__init__(filepath, screenwidth, screenheight, mv_matrix, tanfovx, tanfovy, znear, zfar)

        # Specify shader program to use, and set up OpenGL environment
        glUseProgram(self.program)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Initialise Vertex Array Object; container for all buffers
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Set up vertices to be used each gaussian instance
        self.instance_vertices = np.array([1,-1,1,1,-1,1,-1,-1]).astype(np.float32)
        self.vertex_buffer = glGenBuffers(1)
        self.attribute_vertices = glGetAttribLocation(self.program, "position")
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.instance_vertices.nbytes, self.instance_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_vertices, 2, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_vertices)


        # Per-gaussian properties

        # set up buffer for actual gaussian centers
        self.center_buffer = glGenBuffers(1)
        self.attribute_center = glGetAttribLocation(self.program, "center")
        glBindBuffer(GL_ARRAY_BUFFER, self.center_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.position.nbytes, self.gaussians.position.reshape(-1), GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_center, 3, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_center)
        glVertexAttribDivisor(self.attribute_center, 1)

        # set up rotation buffer
        self.rotation_buffer = glGenBuffers(1)
        self.attribute_rotation = glGetAttribLocation(self.program, "rotation")
        glBindBuffer(GL_ARRAY_BUFFER, self.rotation_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.rotation.nbytes, self.gaussians.rotation.reshape(-1), GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_rotation, 4, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_rotation)
        glVertexAttribDivisor(self.attribute_rotation, 1)

        # set up scale buffer
        self.scale_buffer = glGenBuffers(1)
        self.attribute_scale = glGetAttribLocation(self.program, "scale")
        glBindBuffer(GL_ARRAY_BUFFER, self.scale_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.scale.nbytes, self.gaussians.scale.reshape(-1), GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_scale, 3, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_scale)
        glVertexAttribDivisor(self.attribute_scale, 1)

        # set up colour buffer
        self.colour_buffer = glGenBuffers(1)
        self.attribute_colour = glGetAttribLocation(self.program, "shs")
        glBindBuffer(GL_ARRAY_BUFFER, self.colour_buffer)
        #print(self.gaussians.sh[0])
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.sh.flatten().nbytes, self.gaussians.sh.reshape(-1), GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_colour, 3, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_colour)
        glVertexAttribDivisor(self.attribute_colour, 1)

        # set up opacity buffer
        self.opacity_buffer = glGenBuffers(1)
        self.attribute_opacity = glGetAttribLocation(self.program, "opacity")
        glBindBuffer(GL_ARRAY_BUFFER, self.opacity_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.opacity.nbytes, self.gaussians.opacity, GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_opacity, 1, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_opacity)
        glVertexAttribDivisor(self.attribute_opacity, 1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # set up uniform shader inputs
        self.uniform_projection = glGetUniformLocation(self.program, "projection")
        glUniformMatrix4fv(self.uniform_projection, 1, True, self.projection_matrix)

        self.uniform_view = glGetUniformLocation(self.program, "view")
        glUniformMatrix4fv(self.uniform_view, 1, True, self.modelview_matrix)

        self.uniform_tanfovxy = glGetUniformLocation(self.program, "tanfovxy")
        glUniform2fv(self.uniform_tanfovxy, 1, self.tanfovxy)

        self.uniform_focal = glGetUniformLocation(self.program, "focal")
        glUniform2fv(self.uniform_focal, 1, self.focal)

        self.uniform_viewport = glGetUniformLocation(self.program, "viewport")
        glUniform2fv(self.uniform_viewport, 1, np.array([screenwidth, screenheight]).astype(np.float32))

        # NEW: GLOBAL ROTATION
        self.uniform_globalrotation = glGetUniformLocation(self.program, "global_rotation")
        glUniformMatrix3fv(self.uniform_globalrotation, 1, True, np.identity(3))


    def sort_gaussians(self):
        xyz = np.asarray(self.gaussians.position)
        view_mat = np.asarray(self.modelview_matrix)

        xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
        depth = xyz_view[:, 2, 0]

        indices = np.argsort(depth)

        # sort the center positions, rotations, scales, colours, opacities according to depth
        self.gaussians.position = self.gaussians.position[indices]
        self.gaussians.rotation = self.gaussians.rotation[indices]
        self.gaussians.scale = self.gaussians.scale[indices]
        self.gaussians.sh = self.gaussians.sh[indices]
        self.gaussians.opacity = self.gaussians.opacity[indices]


    def update_buffered_state(self):
        # Refresh buffers of all per-gaussian data, reflecting current depth in scene
        glBindBuffer(GL_ARRAY_BUFFER, self.center_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.position.nbytes, self.gaussians.position.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.rotation_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.rotation.nbytes, self.gaussians.rotation.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.scale_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.scale.nbytes, self.gaussians.scale.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.colour_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.sh.flatten().nbytes, self.gaussians.sh.reshape(-1), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.opacity_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.opacity.nbytes, self.gaussians.opacity, GL_STATIC_DRAW)

    def update_proj(self, proj_matrix):
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        self.projection_matrix = proj_matrix
        glUniformMatrix4fv(self.uniform_projection, 1, GL_TRUE, proj_matrix)

    def update_modelview(self, mv_matrix):
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        self.modelview_matrix = mv_matrix
        glUniformMatrix4fv(self.uniform_view, 1, GL_TRUE, mv_matrix)

    def update_gaussian_rotation(self, global_rotation):
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glUniformMatrix3fv(self.uniform_globalrotation, 1, True, global_rotation)


    def render(self):
        # Clear viewport
        glClearColor(0,0,0,0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Bind to shader program and VAO
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
    
        # Draw each gaussian as an instance of triangle geometry
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, len(self.gaussians.position))
