import numpy as np

# OpenGL commands
from OpenGL.GL import glUseProgram, glHint, glEnable, glBlendFunc,\
    glGenBuffers, glGetAttribLocation, glBindBuffer, glBufferData,\
    glVertexAttribPointer, glEnableVertexAttribArray, glGenVertexArrays,\
    glBindVertexArray, glGetUniformLocation, glUniformMatrix4fv,\
    glClearColor, glClear, glDrawArrays
# OpenGL enums
from OpenGL.GL import GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA,\
    GL_ONE_MINUS_SRC_ALPHA, GL_ARRAY_BUFFER, GL_FLOAT, GL_STATIC_DRAW,\
    GL_COLOR_BUFFER_BIT, GL_NICEST, GL_TRUE, GL_POINT_SMOOTH_HINT,\
    GL_POINT_SMOOTH, GL_VERTEX_PROGRAM_POINT_SIZE, GL_POINTS

from renderer.renderer_template import Renderer


class PointRenderer(Renderer):
    vshader = "./renderer/point_vert.glsl"
    fshader = "./renderer/point_frag.glsl"
    
    def __init__(self, filepath, screenwidth, screenheight, mv_matrix=np.identity(4), fovx=50, fovy=50, znear=0.2, zfar=200):
        tanfovx = np.tan(np.deg2rad(fovx/2))
        tanfovy = np.tan(np.deg2rad(fovy/2))
        super().__init__(filepath, screenwidth, screenheight, mv_matrix, tanfovx, tanfovy, znear, zfar)

        # Use the shader program we linked, set up OpenGL settings
        glUseProgram(self.program)

        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

        # initialise vao
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # set up buffer for actual gaussian centers
        self.center_buffer = glGenBuffers(1)
        self.attribute_center = glGetAttribLocation(self.program, "vCenter")
        
        glBindBuffer(GL_ARRAY_BUFFER, self.center_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.position.nbytes, self.gaussians.position.reshape(-1), GL_STATIC_DRAW)

        glVertexAttribPointer(self.attribute_center, 3, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_center)

        # set up buffer for colours
        self.colour_buffer = glGenBuffers(1)
        self.attribute_colour = glGetAttribLocation(self.program, "vColour")
        glBindBuffer(GL_ARRAY_BUFFER, self.colour_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.sh.flatten().nbytes, self.gaussians.sh.reshape(-1), GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_colour, 3, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_colour)

        # set up opacity buffer
        self.opacity_buffer = glGenBuffers(1)
        self.attribute_opacity = glGetAttribLocation(self.program, "vOpacity")
        glBindBuffer(GL_ARRAY_BUFFER, self.opacity_buffer)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.opacity.nbytes, self.gaussians.opacity, GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_opacity, 1, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_opacity)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # set up mvp matrix
        self.mvp = np.matmul(self.projection_matrix, self.modelview_matrix)
        self.mvp_uniloc = glGetUniformLocation(self.program, "mvp")
        glUniformMatrix4fv(self.mvp_uniloc, 1, GL_TRUE, self.mvp)


    def update_modelview(self, mv_matrix):
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        self.modelview_matrix = mv_matrix
        self.mvp = np.matmul(self.projection_matrix, mv_matrix)
        glUniformMatrix4fv(self.mvp_uniloc, 1, GL_TRUE, self.mvp)

    def update_proj(self, proj_matrix):
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        self.projection_matrix_matrix = proj_matrix
        self.mvp = np.matmul(proj_matrix, self.modelview_matrix)
        glUniformMatrix4fv(self.mvp_uniloc, 1, GL_TRUE, self.mvp)

    def sort_gaussians(self):
        pass

    def update_buffered_state(self):
        pass

    def render(self):
        # Clear viewport to blank
        glClearColor(0,0,0,0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Bind shader program and draw each gaussian as a coloured point
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, len(self.gaussians.position))
