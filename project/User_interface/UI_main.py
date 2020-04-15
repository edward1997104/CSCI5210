from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data.mesh_loader import load_mesh
from User_interface.UI_control import View_Controller
from User_interface.UI_points import Point_Selector
from data.mesh_loader import Mesh
import os


#### SETUP
WINDOWW, WINDOWH = 800, 800
EPS = 1e-5

## model
# mesh_path = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/horse-poses/horse-reference.obj'
mesh_path = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/camel-poses/camel-reference.obj'
output_dir = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/exp_corres/horse_camel'
output_path = os.path.join(output_dir, os.path.basename(mesh_path).split('.')[0]) + '.txt'
print(f"mesh_path : {mesh_path}")
print(f"output_path : {output_path}")
model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(mesh_path)
mesh = Mesh(model_vertices, face_vertices_indcies)
## Pyqt app

# controller
point_selector = Point_Selector(model_vertices, output_path, mesh)
controller = View_Controller(WINDOWW, WINDOWH, point_selector)


# reshape callback
def reshapeFunction(w, h):
    controller.windowH = h
    controller.windowW = w
    glViewport(0, 0, w, h)

## draw
def drawFunc():
    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    ## update matrix
    glMatrixMode(GL_MODELVIEW)

    ## draw points not selected
    glPointSize(2)
    glBegin(GL_POINTS)
    for i in range(model_vertices.shape[0]):
        if point_selector.selected_points[i] == 0:
            glColor3f(1.0, 0.753, 0.796)
            glVertex3f(model_vertices[i, 0], model_vertices[i, 1], model_vertices[i, 2])
    glEnd()


    # draw point selected
    glPointSize(4)
    glBegin(GL_POINTS)
    for i in range(model_vertices.shape[0]):
        if point_selector.selected_points[i] == 1:
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(model_vertices[i, 0], model_vertices[i, 1], model_vertices[i, 2])
            # draw 6 vertices to ensure visual
            glVertex3f(model_vertices[i, 0] + EPS, model_vertices[i, 1], model_vertices[i, 2])
            glVertex3f(model_vertices[i, 0] - EPS, model_vertices[i, 1], model_vertices[i, 2])
            glVertex3f(model_vertices[i, 0], model_vertices[i, 1] + EPS, model_vertices[i, 2])
            glVertex3f(model_vertices[i, 0], model_vertices[i, 1] - EPS, model_vertices[i, 2])
            glVertex3f(model_vertices[i, 0], model_vertices[i, 1], model_vertices[i, 2] + EPS)
            glVertex3f(model_vertices[i, 0], model_vertices[i, 1], model_vertices[i, 2] - EPS)
    glEnd()

    ### draw model
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glBegin(GL_TRIANGLES)
    for i in range(face_vertices_indcies.shape[0]):

        # 1st vertices
        # if point_selector.selected_points[face_vertices_indcies[i, 0]] == 1 or \
        #    point_selector.selected_points[face_vertices_indcies[i, 1]] == 1 or \
        #    point_selector.selected_points[face_vertices_indcies[i, 2]] == 1:
        #     glColor3f(0.0, 0.0, 1.0)
        #     glVertex3f(model_vertices[face_vertices_indcies[i, 0], 0], model_vertices[face_vertices_indcies[i, 0], 1],
        #                model_vertices[face_vertices_indcies[i, 0], 2])
        #     glVertex3f(model_vertices[face_vertices_indcies[i, 1], 0], model_vertices[face_vertices_indcies[i, 1], 1],
        #                model_vertices[face_vertices_indcies[i, 1], 2])
        #     glVertex3f(model_vertices[face_vertices_indcies[i, 2], 0], model_vertices[face_vertices_indcies[i, 2], 1],
        #                model_vertices[face_vertices_indcies[i, 2], 2])
        # else:
        glColor3f(0.8274509804, 0.8274509804, 0.8274509804)
        glVertex3f(model_vertices[face_vertices_indcies[i, 0], 0], model_vertices[face_vertices_indcies[i, 0], 1],
                   model_vertices[face_vertices_indcies[i, 0], 2])
        glColor3f(0.7529411765, 0.7529411765, 0.7529411765)
        glVertex3f(model_vertices[face_vertices_indcies[i, 1], 0], model_vertices[face_vertices_indcies[i, 1], 1],
                   model_vertices[face_vertices_indcies[i, 1], 2])
        glColor3f(0.662745098, 0.662745098, 0.662745098)
        glVertex3f(model_vertices[face_vertices_indcies[i, 2], 0], model_vertices[face_vertices_indcies[i, 2], 1],
                   model_vertices[face_vertices_indcies[i, 2], 2])


    glEnd()
    # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)




    glFlush()

def initializedGL():
    glEnable(GL_DEPTH_TEST)

    ### Loading matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(3, 3, 3, 0, 0, 0, 0, 1, 0)


    ### controler
    controller.reset_projection()

if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(WINDOWW, WINDOWH)
    glutCreateWindow("Mesh marker")
    initializedGL()
    glutReshapeFunc(reshapeFunction)
    glutDisplayFunc(drawFunc)
    # glutIdleFunc(drawFunc)
    glutMouseFunc(controller.button_function)
    glutMotionFunc(controller.motion_function)
    glutKeyboardFunc(controller.keyboard_function)
    glutKeyboardUpFunc(controller.keyboard_up_function)
    # glutMouseWheelFunc(controller.wheel_function)
    glutMainLoop()
