from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math


### SETTING
fovy_rate = 0.01

class View_Controller(object):
    def __init__(self, windowW, windowH, point_selector):
        self.prevMouseX = 0
        self.prevMouseY = 0
        self.pressed_mouse_button = None
        self.mouseModifiers = None
        self.pressed_keyboard_button = None
        self.keyboardModifiers = None
        self.windowW = windowW
        self.windowH = windowH
        self._rotate_scale = 0.5
        self.currFovy = 10
        self._near = 0.01
        self._far = 20
        self.point_selector = point_selector

    def button_function(self, button, state, x, y):

        y = self.windowH - 1 - y

        if state == GLUT_DOWN:
            self.pressed_mouse_button = button
            self.mouseModifiers = glutGetModifiers()
            if button == GLUT_RIGHT_BUTTON:
                self.point_selector.select_point(x, y)

        elif state == GLUT_UP:
            self.pressed_mouse_button = None
            self.mouseModifiers = None

        self.prevMouseX = x
        self.prevMouseY = y

    def motion_function(self, x, y):

        y = self.windowH - 1 - y

        dx = x - self.prevMouseX
        dy = y - self.prevMouseY

        if (dx == 0 and dy == 0):
            return

        # update the code
        self.prevMouseX = x
        self.prevMouseY = y

        if self.pressed_mouse_button == GLUT_LEFT_BUTTON and self.pressed_keyboard_button == b'q':
            tx = 0.01 * dx * self.currFovy / 90.0
            ty = 0.01 * dy * self.currFovy / 90.0

            matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
            glLoadIdentity()
            glTranslated(tx, ty, 0.0)
            glMultMatrixd(matrix)

            # repaint
            glutPostRedisplay()
        elif self.pressed_mouse_button == GLUT_LEFT_BUTTON and self.mouseModifiers == 0:
            nx = -dy
            ny = dx
            scale = math.sqrt(nx*nx + ny* ny)

            nx = nx / scale
            ny = ny / scale
            angle = scale * self._rotate_scale * self.currFovy / 90.0

            ### update code
            matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
            glLoadIdentity()
            glTranslated(matrix[3, 0], matrix[3, 1], matrix[3, 2])
            glRotated(angle, nx, ny, 0.0)
            glTranslated(-matrix[3, 0], -matrix[3, 1], -matrix[3, 2])
            glMultMatrixd(matrix)

            # repaint
            glutPostRedisplay()
        elif self.pressed_mouse_button == GLUT_LEFT_BUTTON and self.mouseModifiers == GLUT_ACTIVE_SHIFT:
            self.update_provy(dy / self.windowH)
            self.reset_projection()
            glutPostRedisplay()

    def reset_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.currFovy, self.windowW / self.windowH, self._near, self._far)
        glMatrixMode(GL_MODELVIEW)

    def update_provy(self, dx):

        _FOVY_K = 0.005

        if self.currFovy < _FOVY_K:
            x = math.log10(self.currFovy) + _FOVY_K - math.log(_FOVY_K)
        else:
            x = self.currFovy

        # add in the x-space
        x += dx * 10

        ##
        if x > 0:
            if x > 179.9:
                x = 179.9
        else:
            x = math.pow(10, x - _FOVY_K + math.log(_FOVY_K))
            x = max(1e-7, x)

        self.currFovy = x

    def keyboard_function(self, key, x, y):

        self.keyboardModifiers = glutGetModifiers()

        if self.keyboardModifiers == GLUT_ACTIVE_CTRL and \
            key == b'\x1a':
            self.point_selector.pop_last_point()
        elif key == b'o':
            self.point_selector.output_selections()

        self.pressed_keyboard_button = key


    def keyboard_up_function(self, key, x, y):
        self.keyboardModifiers = None
        self.pressed_keyboard_button = None



