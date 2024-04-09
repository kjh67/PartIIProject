from OpenGL.GL import *
from renderer import Renderer
from gauss_utils import load_gaussians
import numpy as np


class PointRenderer(Renderer):
    vshader = "./game/point_vert.glsl"
    fshader = "./game/point_frag.glsl"
    
    def __init__(self, filepath, sc_width, sc_height, mvp_matrix):

        # load gaussians from .ply file
        self.gaussians = load_gaussians(filepath)
        self.sc_width = sc_width
        self.sc_height = sc_height

        glViewport(0,0,sc_width,sc_height)

        self.program = glCreateProgram()

        # load and compile shaders
        self.vertex_shader = super().load_shader(GL_VERTEX_SHADER, self.vshader)
        self.fragment_shader = super().load_shader(GL_FRAGMENT_SHADER, self.fshader)

        # create GL program
        glAttachShader(self.program, self.vertex_shader)
        glAttachShader(self.program, self.fragment_shader)
        glLinkProgram(self.program)

        # check that the program linked without errors
        if glGetProgramiv(self.program, GL_LINK_STATUS) == GL_FALSE:
            print("Shader program linking failed")

        # Use the shader program we linked
        glUseProgram(self.program)

        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)

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
        print(self.gaussians.opacity)
        glBufferData(GL_ARRAY_BUFFER, self.gaussians.opacity.nbytes, self.gaussians.opacity, GL_STATIC_DRAW)
        glVertexAttribPointer(self.attribute_opacity, 1, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(self.attribute_opacity)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # set up mvp matrix
        self.mvp = mvp_matrix
        self.mvp_uniloc = glGetUniformLocation(self.program, "mvp")
        glUniformMatrix4fv(self.mvp_uniloc, 1, GL_TRUE, self.mvp)


    def update_mvp(self, mvp_matrix):
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        self.mvp = mvp_matrix
        glUniformMatrix4fv(self.mvp_uniloc, 1, GL_TRUE, mvp_matrix)


    def render(self):

        # clear to a blank screen
        glClearColor(0,0,0,0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # make sure bindings are correctly set up
        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, len(self.gaussians.position))
