from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from tkinter import filedialog
import math
import sys
import os
import numpy as np
from data.mesh_loader import Mesh

class Point_Selector(object):
    def __init__(self, model_vertices, output_path, mesh : Mesh):
        self.model_vertices = model_vertices
        self.distance_thershold = 0.05
        self.selected_points = [0] * model_vertices.shape[0]
        self.selected_points_stack = []
        self.output_path = output_path

        # load the previous file
        if os.path.exists(self.output_path):
            self.selected_points_stack = Point_Selector.load_marker(self.output_path)
            for i in self.selected_points_stack:
                self.selected_points[i] = 1
            glutPostRedisplay()

        self.mesh = mesh

    def select_point(self, windowX, windowY):

        ZDEPTH_WEIGHT = 100

        # near unprojection
        near_coordinate = np.array(gluUnProject(windowX, windowY, 0))
        far_coordinate = np.array(gluUnProject(windowX, windowY, 1))

        sort_temp = []
        for i in range(self.model_vertices.shape[0]):
            calculated_distance = Point_Selector.calculate_distance(near_coordinate, far_coordinate, self.model_vertices[i])
            if calculated_distance < self.distance_thershold and (near_coordinate - far_coordinate).dot(self.mesh.vertices_normals[i] > 0):
                print(f"calculated distance accepted : {calculated_distance}, {self.model_vertices[i]}")
                _, _, z_depth = gluProject(self.model_vertices[i][0], self.model_vertices[i][1],
                                           self.model_vertices[i][2])

                ## hack for z-depth first
                sort_temp.append((z_depth * ZDEPTH_WEIGHT + calculated_distance, i))

        if len(sort_temp) > 0:
            sort_temp = sorted(sort_temp)
            self.selected_points[sort_temp[0][1]] = 1
            self.selected_points_stack.append(sort_temp[0][1])


        glutPostRedisplay()

    def pop_last_point(self):
        if len(self.selected_points_stack) > 0:
            del_index = self.selected_points_stack.pop(-1)
            self.selected_points[del_index] = 0

            glutPostRedisplay()

    def output_selections(self):

        with open(self.output_path, 'w') as f:
            output_index_list = [str(i) for i in self.selected_points_stack]
            output_str = " ".join(output_index_list)
            f.write(output_str)
            f.close()

    @staticmethod
    def calculate_distance(near_coordinate, far_coordinate, point):
        line_vec = far_coordinate - near_coordinate
        near_coord_vec_to_point = point - near_coordinate

        crossed_product = np.cross(line_vec, near_coord_vec_to_point)
        area = np.sqrt(crossed_product.dot(crossed_product))
        line_vec_length = np.sqrt(line_vec.dot(line_vec))
        distance = area / line_vec_length

        return distance

    @staticmethod
    def load_marker(path):
        with open(path, 'r') as f:
            input_str = f.readline()
            f.close()
            return [int(s) for s in input_str.split()]
