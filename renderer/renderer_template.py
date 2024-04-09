#from abc import ABC, abstractmethod
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders

class Renderer:

    def load_shader(self, shader_type, filepath):
        with open(filepath, 'r') as f:
            data = f.read()
        shader = shaders.compileShader(data, shader_type)

        #TODO: check that the shader compiled correctly
        if glGetShaderiv(shader, GL_INFO_LOG_LENGTH) != 0:
            print(f"Error compiling {shader_type}")

        return shader
