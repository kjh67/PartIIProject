import pygame
import numpy as np
import argparse

from OpenGL.GL import glWindowPos2d, glDrawPixels
from OpenGL.GL import GL_RGBA, GL_UNSIGNED_BYTE, GL_CONTEXT_FLAG_DEBUG_BIT_KHR
from threading import Thread, Event as ThreadEvent
from copy import deepcopy

from renderer.gauss_renderer import GaussianRenderer
from renderer.point_renderer import PointRenderer
from game.player_camera import PlayerCamera
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

    renderer = PointRenderer(filename, 800, 600, translation_matrix)
    renderer.render()
    pygame.display.flip()
    xrot = rotate_x(0)
    yrot = rotate_y(0)
    zrot = rotate_z(0)

    while True:
        rotation_matrix = np.matmul(zrot, np.matmul(yrot, xrot))
        mv = np.matmul(translation_matrix, rotation_matrix)
        renderer.update_modelview(mv)
        renderer.render()
        pygame.display.flip()
        pygame.time.wait(30)

        keyspressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_x):
                pygame.quit()
                quit()
        if keyspressed[pygame.K_i]:
            xrot = np.matmul(xrot, rotate_x(0.1))
        if keyspressed[pygame.K_k]:
            xrot = np.matmul(xrot, rotate_x(-0.1))
        if keyspressed[pygame.K_j]:
            yrot = np.matmul(yrot,rotate_y(0.1))
        if keyspressed[pygame.K_l]:
            yrot = np.matmul(yrot,rotate_y(-0.1))

        # only rotations and moving camera to/away from 0,0
        if keyspressed[pygame.K_w]:
            translation_matrix += np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0.1],[0,0,0,0]])
        if keyspressed[pygame.K_s]:
            translation_matrix += np.array([[0,0,0,0],[0,0,0,0],[0,0,0,-0.1],[0,0,0,0]])


def mainloop_gauss(filename, colmap_path=None, points=False):
    screensize = (800, 600)

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
    if colmap_path:
        try:
            model_matrix = generate_model_matrix(colmap_path)
        except FileExistsError:
            model_matrix = np.identity(4)
    else:
        model_matrix = np.identity(4)
    #model_matrix = np.identity(4)

    # initialise renderer
    if not points:
        renderer = GaussianRenderer(filename, *screensize, model_matrix)
    else:
        renderer = PointRenderer(filename, *screensize, model_matrix)

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

    # Instantiate player camera, which will handle all movement and model/view matrices
    camera = PlayerCamera(model_matrix)

    # start the sorting thread before entering main game loop
    sorter.start()

    while True:
        # Check for a completed Gaussian sort, and update renderer state if necessary
        if gaussians_updated.is_set():
            renderer.update_buffered_state()
            gaussians_updated.clear()

        #global_rotation = np.linalg.inv(view_matrix)

        #renderer.update_global_rotation(global_rotation[:3,:3])
        renderer.update_modelview(camera.get_modelview())
        renderer.render()

        # Draw fps counter on the screen
        if display_fps:
            text_surface = display_font.render(f"{str(fps_clock.get_fps())[:5]}", False, (255,255,255,255), (0,0,0, 255))
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2d(0, 0)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # switch double buffered displays
        pygame.display.flip()

        # calculate time since last frame
        t_delta = fps_clock.tick()
        
        # Deal with user input for game settings, quitting etc
        keyspressed = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_x):
                pygame.quit()
                quit()

            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_f):
                display_fps = not display_fps

            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_0):
                renderer.save_frame("./TESTIMAGE.jpg")

        # Update camera position etc with other user input
        camera.update(keyspressed, t_delta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('render_source', type=str, help="Path to .ply file containing splats")
    parser.add_argument('-c', '--colmap_cams_path', type=str, help="Optional path to cameras.bin generated by COLMAP", default=None)
    parser.add_argument('-p', '--points', action='store_true')

    args = parser.parse_args()

    if args.points:
        mainloop_point(args.render_source)
    else:
        mainloop_gauss(args.render_source, args.colmap_cams_path)

    #mainloop_point("C:/Users/kirst/Downloads/point_cloud(12).ply")
    #mainloop_gauss("C:/Users/kirst/Downloads/point_cloud(12).ply", "C:/Users/kirst/Downloads/images.bin")
    #mainloop_gauss("C:/Users/kirst/Downloads/point_cloud(11).ply", "../images.bin")
    #mainloop_gauss("C:/Users/kirst/Downloads/point_cloud(15).ply", "C:/Users/kirst/Downloads/images(1).bin")
