from data.mesh_loader import load_mesh, Mesh
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
from utils.utils import output_obj
import numpy as np
class DeformProblem(object):
    def __init__(self, source_mesh : Mesh, deformed_source_mesh : Mesh, target_mesh : Mesh, corres : np.array):
        self.source_mesh = source_mesh
        self.deformed_source_mesh = deformed_source_mesh
        self.target_mesh = target_mesh
        self.corres = corres

        # calculate deformation of source
        self.calculate_mesh_deformation_matrix()

    def calculate_mesh_deformation_matrix(self):
        self.mesh_deformation_matrices = np.zeros((self.source_mesh.origin_v_matrices.shape[0], 3, 3))

        for i in range(self.source_mesh.origin_v_matrices.shape[0]):
            source_mesh_feature_inv = np.linalg.inv(self.source_mesh.origin_v_matrices[i])
            deformed_source_mesh_feature = self.deformed_source_mesh.origin_v_matrices[i]
            self.mesh_deformation_matrices[i] = deformed_source_mesh_feature.dot(source_mesh_feature_inv)

            # print(f"{i} :{self.mesh_deformation_matrices[i]}")

    def set_up_deform_equation(self):

        ## calculate empty variable
        non_corres_faces_num = np.sum([ len(self.corres[i][self.corres[i] >= 0]) == 0
                                for i in range(self.target_mesh.face_vertices_indcies.shape[0])])
        corres_faces_num = np.sum([ len(self.corres[i][self.corres[i] >= 0])
                                for i in range(self.target_mesh.face_vertices_indcies.shape[0])])

        total_equations_cnt = non_corres_faces_num + corres_faces_num
        # variable
        rows = np.zeros(total_equations_cnt * 9 * 4)
        columns = np.zeros(total_equations_cnt * 9 * 4)
        vals = np.zeros(total_equations_cnt * 9 * 4)

        # constant
        constants = np.zeros(total_equations_cnt * 9)

        equation_cnt = 0
        print("building deformation matrices........")
        for i in range(self.target_mesh.face_vertices_indcies.shape[0]):
            corresind = self.corres[i][self.corres[i] >= 0]
            if len(corresind) > 0:
                for corres_idx in corresind:
                    for axis in range(3):
                        row = np.tile(np.linspace(0, 2, 3) + equation_cnt + axis * 3, [4, 1]).T
                        column = np.tile(self.target_mesh.face_vertices_indcies[i] * 3 + axis, [3, 1])
                        val = self.target_mesh.coefficient_matrices[i]

                        # set variable
                        rows[equation_cnt * 4 + axis * 12 : equation_cnt * 4 + axis * 12 + 12] = row.flatten()
                        columns[equation_cnt * 4 + axis * 12 : equation_cnt * 4 + axis * 12 + 12] = column.flatten()
                        vals[equation_cnt * 4 + axis * 12 : equation_cnt * 4 + axis * 12 + 12] = val.flatten()

                    constant = self.mesh_deformation_matrices[corres_idx]
                    constants[equation_cnt : equation_cnt + 9] = constant.flatten()
                    equation_cnt += 9
            else:
                for axis in range(3):
                    row = np.tile(np.linspace(0, 2, 3) + equation_cnt + axis * 3, [4, 1]).T
                    column = np.tile(self.target_mesh.face_vertices_indcies[i] * 3 + axis, [3, 1])
                    val = self.target_mesh.coefficient_matrices[i]

                    # set variable
                    rows[equation_cnt * 4 + axis * 12: equation_cnt * 4 + axis * 12 + 12] = row.flatten()
                    columns[equation_cnt * 4 + axis * 12: equation_cnt * 4 + axis * 12 + 12] = column.flatten()
                    vals[equation_cnt * 4 + axis * 12: equation_cnt * 4 + axis * 12 + 12] = val.flatten()

                constant = np.eye(3)
                constants[equation_cnt: equation_cnt + 9] = constant.flatten()
                equation_cnt += 9


        deform_matrix = coo_matrix((vals, (rows, columns)), shape = (total_equations_cnt * 9,
                                                                     self.target_mesh.vertices.shape[0] * 3))
        print("Done building deformation matrices........")

        print(f"constants sum : {np.sum(constants)}")
        return deform_matrix, constants





if __name__ == "__main__":

    # setting
    for i in range(2, 11):
        pose_idx = i
        source_mesh_path = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/horse-poses/horse-reference.obj'
        deformed_source_mesh_path = f'/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/horse-poses/horse-{"0" + str(pose_idx) if pose_idx < 10 else pose_idx}.obj'
        target_mesh_path = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/camel-poses/camel-reference.obj'
        corres_path = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/result/horse_camel/corres/target_to_source_corres.npy'
        output_file_path = f'/Users/edwardhui/Desktop/previous_file/CSCI5210/project/result/horse_camel/output_mesh/{pose_idx}.obj'

        ## loading meshes
        model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(source_mesh_path)
        source_mesh = Mesh(model_vertices, face_vertices_indcies)

        model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(deformed_source_mesh_path)
        deformed_source_mesh = Mesh(model_vertices, face_vertices_indcies)

        model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(target_mesh_path)
        target_mesh = Mesh(model_vertices, face_vertices_indcies)

        ## load deform
        corres = np.load(corres_path)
        # corres = np.arange(0, source_mesh.face_vertices_indcies.shape[0])[:, np.newaxis] # debug

        deform_problem = DeformProblem(source_mesh, deformed_source_mesh, target_mesh, corres)

        A, b = deform_problem.set_up_deform_equation()

        print(f"start solving deformation equations....., A: {A.shape}, b : {b.shape}")
        result = lsqr(A, b, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
        print(f"result : {result[0]}")
        deformed_target_mesh_vertices = np.reshape(result[0], [-1, 3])[:deform_problem.target_mesh.real_vertices_num, :]
        deformed_target_mesh = Mesh(deformed_target_mesh_vertices, deform_problem.target_mesh.face_vertices_indcies, build_triangle_features = False)
        output_obj(output_file_path, deformed_target_mesh)
        print(f"done solving deformation equations.......")