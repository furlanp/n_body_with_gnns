import pygame as pg
import numpy as np
import numpy.linalg as la
from OpenGL.GL import *
from OpenGL.GLU import *

class PygApp:
    def __init__(self, update_func): 
        
        self.ticks = 60
        self.update_func = update_func
        self.display = [800, 600]
        self.display_cen = [400, 300]
        self.paused = False
        self.running = True

        # pygame init 
        pg.init()
        pg.display.set_mode(self.display, pg.OPENGL|pg.DOUBLEBUF)
        pg.mouse.set_visible(False)
        self.clock = pg.time.Clock()

        # camera settings
        self.camera_pos = np.array([0.0, 0.0, 3.0])
        self.camera_front = np.array([0.0, 0.0, -1.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])
        self.camera_speed = 0.1

        self.fov = 45.0
        self.yaw = -90.0
        self.pitch = 0.0
        self.mouse_move = [0, 0]
        self.mouse_last = self.display_cen
        self.sensitivity = 0.1

        # opengl config
        glEnable(GL_DEPTH_TEST)
        glPointSize(5.0)
        
        self.run()

    def run(self):
        while self.running:
            # mouse and keyboard interaction
            self.process_input()

            if self.paused:   # skip if paused
                continue

            # make sure that mouse stays inside window
            self.mouse_last = self.display_cen
            pg.mouse.set_pos(self.mouse_last)

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(self.fov, (self.display[0] / self.display[1]), 0.1, 100.0)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(*self.camera_pos, *(self.camera_pos + self.camera_front), *self.camera_up)

            self.draw_points()

            pg.display.flip()
        
            # timing
            self.clock.tick(self.ticks)
    
    def process_input(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    self.paused = not self.paused
                    pg.mouse.set_visible(self.paused) 
            
            if event.type == pg.MOUSEMOTION:
                self.mouse_move = [event.pos[0] - self.mouse_last[0], self.mouse_last[1] - event.pos[1]] 
                self.mouse_last = [event.pos[i] for i in range(2)]
                
                self.yaw += self.mouse_move[0] * self.sensitivity
                self.pitch += self.mouse_move[1] * self.sensitivity

                # make sure that pitch is inside [-89.0, 89.0]
                self.pitch = max(min(self.pitch, 89.0), -89.0)
                
                front = [0, 0, 0]
                front[0] = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
                front[1] = np.sin(np.radians(self.pitch))
                front[2] = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
                self.camera_front = np.array(front) / la.norm(front)

            if event.type == pg.MOUSEWHEEL:
                self.fov += event.y
                
                # make sure that fov is inside [1.0, 45.0]
                self.fov = max(min(self.fov, 45.0), 1.0)

        if self.paused:   # skip if paused
            return
        
        keys = pg.key.get_pressed()

        if keys[pg.K_w]:
            self.camera_pos += self.camera_front * self.camera_speed
        if keys[pg.K_s]:
            self.camera_pos -= self.camera_front * self.camera_speed 
        if keys[pg.K_d]:
            v = np.cross(self.camera_front, self.camera_up)
            self.camera_pos += v / la.norm(v) * self.camera_speed
        if keys[pg.K_a]:
            v = np.cross(self.camera_front, self.camera_up)
            self.camera_pos -= v / la.norm(v) * self.camera_speed

    def draw_points(self):
        glBegin(GL_POINTS)
        for point in self.update_func():
            glColor3f(1.0, 0, 0)
            glVertex3f(point[0], point[1], point[2])
        glEnd()

    def quit(self):
        pg.quit()


# testing
if __name__ == '__main__':
    
    def update_func():
        return [[0, 0, 0]]

    app = PygApp(update_func)