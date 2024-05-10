from abc import ABC, abstractmethod
import numpy as np
import cv2
import OpenGL.GL.shaders as shaders

# OpenGL commands
from OpenGL.GL import glViewport, glCreateProgram, glAttachShader,\
    glLinkProgram, glGetProgramiv, glGetShaderiv, glReadPixels
# OpenGL enums
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_LINK_STATUS,\
    GL_FALSE, GL_INFO_LOG_LENGTH, GL_BGR, GL_UNSIGNED_BYTE

from renderer.gauss_utils import Gaussians


class Renderer(ABC):
    """Abstract base class for renderers operating on Gaussian data.
    
    Contains some standard OpenGL setup in __init__, and otherwise defines
    the abstract methods which must be implemented by renderer classes.
    """

    vshader: str
    fshader:str

    def __init__(self, filepath, screenwidth, screenheight, mv_matrix, tanfovx, tanfovy, znear, zfar):
        """Initialises renderer attributes
        
        Stores initial camera intrinsics, and loads gaussian information.
        Also performs shader compilation and linking, and sets up the OpenGL
        viewport. Does not initialise any buffers or buffer objects.
        """
    
        self.gaussians = Gaussians.load_gaussians(filepath)

        self.screenwidth = screenwidth
        self.screenheight = screenheight

        self.modelview_matrix = mv_matrix

        top = tanfovx * znear
        bottom = -top
        right = tanfovy * znear
        left = -right

        self.projection_matrix = np.array([
            [2 * znear / (right - left), 0, (right+left)/(right-left), 0],
            [0, 2 * znear / (top - bottom), (top + bottom) / (top - bottom), 0],
            [0, 0, -zfar / (zfar-znear), -(zfar*znear) / (zfar-znear)],
            [0, 0, -1, 0]
        ])

        self.tanfovxy = np.array([tanfovx, tanfovy]).astype(np.float32)
        self.focal = np.array([screenwidth/(2*tanfovx), screenheight/(2*tanfovy)]).astype(np.float32)

        glViewport(0,0,screenwidth,screenheight)

        self.vertex_shader = self._load_shader(GL_VERTEX_SHADER, self.vshader)
        self.fragment_shader = self._load_shader(GL_FRAGMENT_SHADER, self.fshader)

        self.program = glCreateProgram()
        glAttachShader(self.program, self.vertex_shader)
        glAttachShader(self.program, self.fragment_shader)
        glLinkProgram(self.program)

        if glGetProgramiv(self.program, GL_LINK_STATUS) == GL_FALSE:
            print("Shader program linking failed")
            quit(code=1)


    def _load_shader(self, shader_type, filepath):
        with open(filepath, 'r') as f:
            data = f.read()
        shader = shaders.compileShader(data, shader_type)

        #TODO: check that the shader compiled correctly
        if glGetShaderiv(shader, GL_INFO_LOG_LENGTH) != 0:
            print(f"Error compiling {shader_type}")

        return shader
    

    def get_frame(self):
        """Retrieves the current frame from the framebuffer"""
        # glReadPixels returns the current contents of the frame buffer as a string
        imagedata = glReadPixels(0, 0, self.screenwidth, self.screenheight, GL_BGR, GL_UNSIGNED_BYTE)
        image = np.asarray(np.fromstring(imagedata, np.uint8).reshape(self.screenheight, self.screenwidth, 3))
        # Since OpenGL reads the image from the bottom left, but cv2 writes from the top left,
        # we need to flip the image vertically
        image = cv2.flip(image, 0)
        return image
    
    def save_frame(self, filename):
        image = self.get_frame()
        cv2.imwrite(filename, image)
        cv2.imshow("Image", image)
    
    def save_video(self, filename, fps, cam_poses_per_frame):
        video = cv2.VideoWriter(filename, -1, fps, (self.screenwidth, self.screenheight))
        for pose in cam_poses_per_frame:
            self.update_modelview(pose)
            self.sort_gaussians()
            self.update_buffered_state()
            self.render()
            frame = self.get_frame()
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()        

    @abstractmethod
    def update_modelview(self, mv_matrix):
        pass

    @abstractmethod
    def update_proj(self, proj_matrix):
        pass

    @abstractmethod
    def update_buffered_state(self):
        pass

    @abstractmethod
    def sort_gaussians(self):
        pass

    @abstractmethod
    def render(self):
        pass
