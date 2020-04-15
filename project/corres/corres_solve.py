import numpy as np
from data.mesh_loader import Mesh, load_mesh
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
import os

from utils.utils import output_obj


class CorresProblem(object):
    def __init__(self, source_mesh : Mesh, target_mesh : Mesh, marker_pair):
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.marker_pair = marker_pair

    def set_up_phrase_1_equation(self, Ws, Wi, MARKER_WEIGHT):

        n_adj = np.sum([len(self.source_mesh.get_adj_face(i)) for i in range(self.source_mesh.face_vertices_indcies.shape[0])])

        ## smoothness equations
        rows_smoothness = np.zeros(9*n_adj*4)
        columns_smoothness_center = np.zeros(9*n_adj*4)
        vals_smoothness_center = np.zeros(9*n_adj*4)
        columns_smoothness_adj = np.zeros(9*n_adj*4)
        vals_smoothness_adj = np.zeros(9*n_adj*4)

        ## smoothness constant
        smoothness_constant = np.zeros(9*n_adj)

        # set up equation for smoothness
        print("building smoothness......")
        equation_cnt = 0
        for i in range(self.source_mesh.face_vertices_indcies.shape[0]):
            for j in self.source_mesh.get_adj_face(i):

                # check in marker
                marker_flags = np.full((2, 4) , -1)
                for idx in range(0, 3):
                    if self.source_mesh.face_vertices_indcies[i][idx] in self.marker_pair[0]:
                        pair_index = self.marker_pair[0].index(self.source_mesh.face_vertices_indcies[i][idx])
                        marker_flags[0, idx] = self.marker_pair[1][pair_index]
                    if self.source_mesh.face_vertices_indcies[j][idx] in self.marker_pair[0]:
                        pair_index = self.marker_pair[0].index(self.source_mesh.face_vertices_indcies[j][idx])
                        marker_flags[1, idx] = self.marker_pair[1][pair_index]

                # for x, y, z direction
                for axis in range(3):

                    row = np.tile(np.linspace(0, 2, 3, dtype = np.int32) + equation_cnt + axis * 3, [4, 1]).T # 3 * 4
                    ## data for center
                    column_center = np.tile(self.source_mesh.face_vertices_indcies[i] * 3 + axis, [3, 1]) # 3 * 4
                    val_center = Ws * self.source_mesh.coefficient_matrices[i]

                    ## if marker exist
                    # if np.sum(marker_flags[0, :]) > -4:
                    #     constant_vector = val
                    #     smoothness_constant[equation_cnt + axis * 3 : equation_cnt + axis * 3 + 3] =\
                    #         smoothness_constant[equation_cnt + axis * 3 : equation_cnt + axis * 3 + 3]

                    ## data for adj
                    column_adj = np.tile(self.source_mesh.face_vertices_indcies[j] * 3 + axis, [3, 1])  # 3 * 4
                    val_adj = - Ws * self.source_mesh.coefficient_matrices[j]

                    ## keep record
                    rows_smoothness[equation_cnt * 4 + axis * 3 * 4: equation_cnt * 4 + axis * 3 * 4 + 12] = \
                        row.flatten()
                    columns_smoothness_center[equation_cnt * 4 + axis * 3 * 4: equation_cnt * 4 + axis * 3 * 4 + 12]  = \
                        column_center.flatten()
                    vals_smoothness_center[equation_cnt * 4 + axis * 3 * 4: equation_cnt * 4 + axis * 3 * 4 + 12]  = \
                        val_center.flatten()
                    columns_smoothness_adj[equation_cnt * 4 + axis * 3 * 4: equation_cnt * 4 + axis * 3 * 4 + 12]  = \
                        column_adj.flatten()
                    vals_smoothness_adj[equation_cnt * 4 + axis * 3 * 4: equation_cnt * 4 + axis * 3 * 4 + 12]  = \
                        val_adj.flatten()


                # 9 equations are set up
                equation_cnt += 9

        # Smoothness matrices
        smoothness_1_M = coo_matrix((vals_smoothness_center, (rows_smoothness, columns_smoothness_center)),
                                    shape = (9 * n_adj, 3 * self.source_mesh.vertices.shape[0]))
        smoothness_2_M = coo_matrix((vals_smoothness_adj, (rows_smoothness, columns_smoothness_adj)),
                                    shape = (9 * n_adj, 3 * self.source_mesh.vertices.shape[0]))

        smoothness_matrix = smoothness_1_M + smoothness_2_M

        print("done building smoothness......")

        ## identity
        print("building identity......")

        # equation for rows / columns / vals for left equation
        rows_identity = np.zeros(9*self.source_mesh.face_vertices_indcies.shape[0]*4)
        columns_identity = np.zeros(9*self.source_mesh.face_vertices_indcies.shape[0]*4)
        vals_identity = np.zeros(9*self.source_mesh.face_vertices_indcies.shape[0]*4)

        # constant
        identity_constant = Wi * np.tile(np.eye(3).flatten(), [self.source_mesh.face_vertices_indcies.shape[0]])

        identity_equation_cnt = 0
        for i in range(self.source_mesh.face_vertices_indcies.shape[0]):

            for axis in range(3):
                row = np.tile(np.linspace(0, 2, 3, dtype = np.int32) + identity_equation_cnt + axis * 3, [4, 1]).T # 3 * 4

                ## data for center
                column_identity = np.tile(self.source_mesh.face_vertices_indcies[i] * 3 + axis, [3, 1])  # 3 * 4
                val_identity = Wi * self.source_mesh.coefficient_matrices[i]

                ## keep record
                rows_identity[identity_equation_cnt * 4 + axis * 3 * 4: identity_equation_cnt * 4 + axis * 3 * 4 + 12] = \
                    row.flatten()
                columns_identity[identity_equation_cnt * 4 + axis * 3 * 4: identity_equation_cnt * 4 + axis * 3 * 4 + 12] = \
                    column_identity.flatten()
                vals_identity[identity_equation_cnt * 4 + axis * 3 * 4: identity_equation_cnt * 4 + axis * 3 * 4 + 12] = \
                    val_identity.flatten()

            identity_equation_cnt += 9

        # identity matrix
        identity_matrix = coo_matrix((vals_identity, (rows_identity, columns_identity)),
                                   shape = (self.source_mesh.face_vertices_indcies.shape[0] * 9, 3 * self.source_mesh.vertices.shape[0]))
        print("done building identity......")

        print("building marker......")
        # Trial on fixing marker row vertices
        marker_rows = np.zeros(len(self.marker_pair[0]) * 3)
        marker_columns = np.zeros(len(self.marker_pair[0]) * 3)
        marker_vals = np.zeros(len(self.marker_pair[0]) * 3)
        marker_targets = np.zeros(len(self.marker_pair[0]) * 3)
        marker_equation_cnt = 0

        for i in range(len(self.marker_pair[0])):
            row = np.linspace(0, 2, 3, dtype = np.int32) + marker_equation_cnt # 3 x 1
            column = np.linspace(0, 2, 3, dtype = np.int32) + self.marker_pair[0][i] * 3
            target_val = np.array([
                self.target_mesh.vertices[self.marker_pair[1][i]][0], #x
                self.target_mesh.vertices[self.marker_pair[1][i]][1], #y
                self.target_mesh.vertices[self.marker_pair[1][i]][2], #z
            ])

            print(f'target_val : {target_val}')
            marker_rows[marker_equation_cnt: marker_equation_cnt + 3] = row.flatten()
            marker_columns[marker_equation_cnt: marker_equation_cnt + 3] = column.flatten()
            marker_vals[marker_equation_cnt: marker_equation_cnt + 3] = np.ones(3).flatten() * MARKER_WEIGHT
            marker_targets[marker_equation_cnt: marker_equation_cnt + 3] = target_val.flatten() * MARKER_WEIGHT

            marker_equation_cnt += 3

        # marker matrix
        marker_matrix = coo_matrix((marker_vals, (marker_rows, marker_columns)),
                                   shape = (len(self.marker_pair[0]) * 3, 3 * self.source_mesh.vertices.shape[0]))

        print("done building marker......")
        print(smoothness_constant.shape, marker_targets.shape)
        final_matrix = vstack((smoothness_matrix, identity_matrix, marker_matrix))
        final_constant = np.concatenate((smoothness_constant, identity_constant, marker_targets))
        return final_matrix, final_constant

    def set_up_phrase_2_equation(self, deformed_source_mesh : Mesh, Wc):

        # set up the variables
        print("building phrase 2......")
        rows = np.arange(0, deformed_source_mesh.real_vertices_num * 3)
        columns = np.arange(0, deformed_source_mesh.real_vertices_num * 3)
        vals_one = Wc * np.ones(deformed_source_mesh.real_vertices_num * 3)

        # constant
        constant = np.zeros(deformed_source_mesh.real_vertices_num * 3)
        for i in range(deformed_source_mesh.real_vertices_num):

            if i in self.marker_pair[0]:
                index_of_maker = self.marker_pair[0].index(i)
                constant[i * 3 : i * 3 + 3] = Wc *np.array([
                self.target_mesh.vertices[self.marker_pair[1][index_of_maker]][0], #x
                self.target_mesh.vertices[self.marker_pair[1][index_of_maker]][1], #y
                self.target_mesh.vertices[self.marker_pair[1][index_of_maker]][2], #z
            ])
            else:
                closest_idx = CorresProblem.find_closest_valid_pt(deformed_source_mesh, self.target_mesh, i)
                constant[i * 3 : i * 3 + 3] = Wc * np.array( [
                        self.target_mesh.vertices[closest_idx][0], #x
                        self.target_mesh.vertices[closest_idx][1], #y
                        self.target_mesh.vertices[closest_idx][2], #z
                ])

        # phrase_2 matrix
        phrase_2_matrix = coo_matrix((vals_one, (rows, columns)),
                                     shape = (deformed_source_mesh.real_vertices_num * 3, 3 * self.source_mesh.vertices.shape[0]))

        print("done building phrase 2......")
        return phrase_2_matrix, constant


    @staticmethod
    def find_closest_valid_pt(source_mesh : Mesh, target_mesh : Mesh, source_mesh_index):

        sq_distance = (np.tile(source_mesh.vertices[source_mesh_index], [target_mesh.real_vertices_num, 1]) - target_mesh.vertices[:target_mesh.real_vertices_num, :]) ** 2
        sum_sq_distance = np.sum(sq_distance, axis = 1)

        sorted_idx = np.argsort(sum_sq_distance)

        for i in sorted_idx:
            if source_mesh.vertices_normals[source_mesh_index].dot(target_mesh.vertices_normals[i]) > 0:
                return i

        return None


def load_marker(path):
    with open(path, 'r') as f:
        input_str = f.readline()
        f.close()
        return [int(s) for s in input_str.split()]


if __name__ == "__main__":

    #### SETTING
    mesh_path_horse = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/horse-poses/horse-reference.obj'
    marker_path_horse = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/exp_corres/horse_camel/horse-reference.txt'
    mesh_path_camel = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/camel-poses/camel-reference.obj'
    marker_path_camel = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/exp_corres/horse_camel/camel-reference.txt'
    output_file_path = './testing_more.obj'
    phrase_one = True
    phrase_two = True
    Ws = 1.0
    Wi = 0.0001
    MARKER_WEIGHT = 10000.0
    phrase_2_trial = 4
    Wcs = [1.0, 10.0, 100.0, 300.0]

    ## loading meshes
    model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(mesh_path_horse)
    mesh_horse = Mesh(model_vertices, face_vertices_indcies)
    marker_horse = load_marker(marker_path_horse)

    model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(mesh_path_camel)
    mesh_camel = Mesh(model_vertices, face_vertices_indcies)
    marker_camel = load_marker(marker_path_camel)

    # print(marker_horse, marker_camel)
    assert len(marker_horse) == len(marker_camel)
    problem = CorresProblem(mesh_horse, mesh_camel, [marker_horse, marker_camel])

    A, b = problem.set_up_phrase_1_equation(Ws = Ws, Wi = Wi, MARKER_WEIGHT= MARKER_WEIGHT)


    if not os.path.exists(output_file_path) or phrase_one:
        print(f"start solving first phrase equations....., A: {A.shape}, b : {b.shape}")
        result = lsqr(A, b, iter_lim= 30000, atol = 1e-8, btol=1e-8, conlim=1e7, show = False)
        deformed_mesh_vertices_phrase_1 = np.reshape(result[0], [-1, 3])[:problem.source_mesh.real_vertices_num, :]
        deformed_mesh = Mesh(deformed_mesh_vertices_phrase_1, problem.source_mesh.face_vertices_indcies)
        output_obj(os.path.basename(output_file_path).split('.')[0] + f'_pharse_1.obj', deformed_mesh)
        print(f"done solving first phrase equations.......")
    else:
        model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(output_file_path)
        deformed_mesh = Mesh(model_vertices, face_vertices_indcies)
        print(f"deformed_mesh normal shape : {deformed_mesh.vertices_normals.shape}")
        print(f"target normal shape : {problem.target_mesh.vertices_normals.shape}")
        print(f"target vertices shape : {problem.target_mesh.vertices.shape}")


    if phrase_two:
        for i in range(phrase_2_trial):
            A_2, b_2 = problem.set_up_phrase_2_equation(deformed_mesh, Wcs[i])

            A_phrase_2 = vstack((A, A_2))
            b_phrase_2 = np.concatenate((b, b_2))

            print(f"start solving second phrase 2 equations {i} trial....., A: {A_phrase_2.shape}, b : {b_phrase_2.shape}")
            result = lsqr(A_phrase_2, b_phrase_2, iter_lim=10000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
            deformed_mesh_vertices_phrase_2 = np.reshape(result[0], [-1, 3])[:problem.source_mesh.real_vertices_num, :]
            deformed_mesh = Mesh(deformed_mesh_vertices_phrase_2, problem.source_mesh.face_vertices_indcies)
            output_obj(os.path.basename(output_file_path).split('.')[0] + f'_pharse_2_{i}.obj', deformed_mesh)
            print(f"done solving first phrase 2 equations {i} trial.......")


    # index = CorresProblem.find_closest_valid_pt(mesh_horse, mesh_camel, 0)
    #
    # print(f"source mesh pt : {mesh_horse.vertices[0]} normal : {mesh_horse.vertices_normals[0]}")
    # print(f"target mesh pt : {mesh_camel.vertices[index]} normal : {mesh_horse.vertices_normals[index]}")
    # print(f"dot products : {mesh_camel.vertices_normals[index].dot(mesh_horse.vertices[0])}")