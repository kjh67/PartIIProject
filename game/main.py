import pygame
from OpenGL.GL import *
import numpy as np
import os
from threading import Thread
from threading import Event as ThreadEvent

from renderer.gauss_renderer import GaussianRenderer
from renderer.point_renderer import PointRenderer

from game.colmap_data_utils import generate_model_matrix

def rotate_x(theta):
    return np.array([[1,0,0,0],
                    [0,np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0,0,0,1]])

def rotate_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                     [0,1,0,0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0,0,0,1]])

def rotate_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0,0,1,0],
                    [0,0,0,1]])

def mainloop_point(filename):
    pygame.init()
    pygame.display.gl_set_attribute(GL_CONTEXT_FLAG_DEBUG_BIT_KHR, True)
    # also adding the pygame.DOUBLEBUF flag breaks everything
    pygame.display.set_mode((800,600), pygame.OPENGL)
    pygame.display.set_caption("Splatting Point viewer")

    # translate -5 in x
    translation_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-3],[0,0,0,1]]).astype(np.float32)

    znear = 0.2
    zfar = 200

    w = 2*0.11547
    h = 2*0.0866
    projection_matrix = np.array([[2*znear/w,0,0,0],[0,2*znear/h,0,0],[0,0,-((zfar+znear)/(zfar-znear)),-2*((zfar*znear)/(zfar-znear))],[0,0,-1,0]])
    
    
    # for this equation: h and w are of the actual canvas, fx and fy are camera FOVs (div by 2)
    # w = 800
    # h = 600
    # fy = 724
    # fx = 692
    # projection_matrix = np.array([[2*fx/w, 0, 0, 0],[0, 2*fy/h,0,0],[0,0,zfar/(zfar-znear),1],[0,0,-(zfar*znear)/(zfar-znear),0]])

    # basic proj matrix, cancels out z
    #projection_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]])

    mvp = np.matmul(projection_matrix, translation_matrix).astype(np.float32)
    #mvp = view_matrix


    renderer = PointRenderer(filename, 800, 600, mvp)
    renderer.render()
    pygame.display.flip()
    pygame.time.wait(30)
    xrot = rotate_x(0)
    yrot = rotate_y(0)
    zrot = rotate_z(0)

    while True:
        rotation_matrix = np.matmul(zrot, np.matmul(yrot, xrot))
        mvp = np.matmul(projection_matrix, np.matmul(translation_matrix, rotation_matrix))
        renderer.update_mvp(mvp)
        renderer.render()
        pygame.display.flip()
        pygame.time.wait(30)

        keyspressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_x):
                pygame.quit()
                quit()
        if keyspressed[pygame.K_i]:
            xrot = np.matmul(xrot, rotate_x(5))
        if keyspressed[pygame.K_k]:
            xrot = np.matmul(xrot, rotate_x(-5))
        if keyspressed[pygame.K_j]:
            yrot = np.matmul(yrot,rotate_y(5))
        if keyspressed[pygame.K_l]:
            yrot = np.matmul(yrot,rotate_y(-5))

        # only rotations and moving camera to/away from 0,0
        if keyspressed[pygame.K_w]:
            translation_matrix += np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]])
        if keyspressed[pygame.K_s]:
            translation_matrix += np.array([[0,0,0,0],[0,0,0,0],[0,0,0,-0.1],[0,0,0,0]])



def mainloop_gauss(filename, colmap_path):
    screensize = (800,600)

    # pygame setup
    pygame.init()
    pygame.display.gl_set_attribute(GL_CONTEXT_FLAG_DEBUG_BIT_KHR, True)
    # before was pygame.DOUBLEBUF|pygame.OPENGL
    pyg_screen = pygame.display.set_mode(screensize, pygame.OPENGL)
    pygame.display.set_caption("Splatting viewer")

    # initialise fps display font
    display_font = pygame.font.SysFont("", 25)
    display_fps = False

    # attempt to generate model matrix from COLMAP data
    try:
        model_matrix = generate_model_matrix(os.path.join(colmap_path, "images.bin"))
    except FileExistsError:
        model_matrix = np.identity(4)

    # get projection matrix (TODO: MOVE TO RENDERER)
    znear = 0.2
    zfar = 200
    w = 2*0.11547
    h = 2*0.0866
    # htany = np.tan(np.deg2rad(22.5))
    # htanx = htany * w / h
    #projection_matrix = np.array([[1/htanx,0,0,0],[0,1/htany,0,0],[0,0,(zfar/(zfar-znear)),-2*((zfar*znear)/(zfar-znear))],[0,0,-1,0]])
    projection_matrix = np.array([[2*znear/w,0,0,0],[0,2*znear/h,0,0],[0,0,-((zfar+znear)/(zfar-znear)),-2*((zfar*znear)/(zfar-znear))],[0,0,-1,0]])

    # initialise renderer
    renderer = GaussianRenderer(filename, *screensize, model_matrix, projection_matrix)

    # set up gaussian sorting in the background - use event to signal to the main loop when a sort has been completed
    gaussians_updated = ThreadEvent()
    gaussians_updated.clear()
    def sorting_loop(gaussians_updated):
        while True:
            renderer.sort_gaussians()
            gaussians_updated.set()
    sorter = Thread(target=sorting_loop, args=[gaussians_updated], daemon=True)

    # set up clock for movement and fps calculations
    fps_clock = pygame.time.Clock()
    fps_clock.tick()

    # set up matrices
    translation_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.float32)
    model_rotation_matrix = np.identity(4)
    view_rotation_matrix = np.identity(4)
    view_matrix = np.identity(4)
    modelview_matrix = np.matmul(translation_matrix, model_matrix)

    # start the sorting thread before entering main loop
    sorter.start()

    while True:
        # check for a completed sort, update renderer state if necessary
        if gaussians_updated.is_set():
            renderer.update_buffered_state()
            gaussians_updated.clear()
        
        # compute transformation matrices, do rendering
        model_matrix_delta = np.matmul(model_rotation_matrix, translation_matrix)
        model_matrix = np.matmul(model_matrix_delta, model_matrix)
        view_matrix = np.matmul(view_rotation_matrix, view_matrix)
        modelview_matrix = np.matmul(view_matrix, model_matrix)
        renderer.update_modelview(modelview_matrix)
        renderer.render()

        # draw fps counter on the screen (white text on black background)
        if display_fps:
            text_surface = display_font.render(f"{str(fps_clock.get_fps())[:5]}", False, (255,255,255,255), (0,0,0, 255))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2d(0, 0)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # switch double buffered displays
        pygame.display.flip()

        # calculate time since last frame, and print fps
        t_delta = fps_clock.tick()
        #t_delta = 10

        # take user input
        keyspressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_x):
                pygame.quit()
                quit()

            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_f):
                display_fps = not display_fps


        # # x axis is across the screen
        # if keyspressed[pygame.K_i]:
        #     xrot = np.matmul(xrot, rotate_x(t_delta*np.pi/2000))
        # if keyspressed[pygame.K_k]:
        #     xrot = np.matmul(xrot, rotate_x(t_delta*np.pi/2000))
        # y axis is vertical

        if keyspressed[pygame.K_i]:
            view_rotation_matrix = rotate_x(t_delta/1500)
        elif keyspressed[pygame.K_k]:
            view_rotation_matrix = rotate_x(-t_delta/1500)
        else:
            view_rotation_matrix = np.identity(4)

        # turning left/right
        if keyspressed[pygame.K_j]:
            model_rotation_matrix = rotate_y(t_delta/1500)
        elif keyspressed[pygame.K_l]:
            model_rotation_matrix = rotate_y(-t_delta/1500)
        else:
            model_rotation_matrix = np.identity(4)
            
        # forwards/backwards movement
        if keyspressed[pygame.K_w]:
            translation_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.001*t_delta],[0,0,0,1]])
        elif keyspressed[pygame.K_s]:
            translation_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-0.001*t_delta],[0,0,0,1]])
        else:
            translation_matrix = np.identity(4)


if __name__ == "__main__":
    #mainloop_point("C:/Users/kirst/Downloads/point_cloud(7).ply")
    mainloop_gauss("C:/Users/kirst/Downloads/point_cloud(7).ply", "..")
