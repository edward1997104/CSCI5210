from data.mesh_loader import load_mesh, Mesh
from scipy.spatial import KDTree
import os
import numpy as np
class CorresBuilder(object):

    def __init__(self, deformed_source_mesh : Mesh, target_mesh : Mesh):
        self.deformed_source_mesh = deformed_source_mesh
        self.target_mesh = target_mesh

    def build_corres(self, Max_Closest_Point, dis_thershold):

        # build KD-tree
        source_centeroid_KD = KDTree(self.deformed_source_mesh.faces_centeroid)
        target_centeroid_KD = KDTree(self.target_mesh.faces_centeroid)

        source_to_target_corres =  CorresBuilder.find_correspondances(
            source_mesh = self.deformed_source_mesh,
            target_mesh = self.target_mesh,
            target_centeroid_KD = target_centeroid_KD,
            Max_Closest_Point = Max_Closest_Point,
            dis_thershold = dis_thershold
        )

        target_to_source_corres =  CorresBuilder.find_correspondances(
            source_mesh = self.target_mesh,
            target_mesh = self.deformed_source_mesh,
            target_centeroid_KD = source_centeroid_KD,
            Max_Closest_Point = Max_Closest_Point,
            dis_thershold = dis_thershold
        )

        return source_to_target_corres, target_to_source_corres

    @staticmethod
    def find_correspondances(source_mesh : Mesh, target_mesh : Mesh, target_centeroid_KD : KDTree, Max_Closest_Point, dis_thershold):

        # output should be a Number of source face X Max_Closest_Point matrix
        output_corres = np.full((source_mesh.faces_centeroid.shape[0], Max_Closest_Point), -1)

        for i in range(source_mesh.faces_centeroid.shape[0]):

            _, corresind = target_centeroid_KD.query(source_mesh.faces_centeroid[i], k = Max_Closest_Point,
                                                     distance_upper_bound = dis_thershold)
            # filter not matching point
            corresind = corresind[corresind >= 0]
            corresind = corresind[corresind < target_mesh.faces_centeroid.shape[0]]

            # check normal direction match
            cos_products = np.sum(
                np.tile(source_mesh.face_normals[i], [len(corresind), 1]) *
                                  target_mesh.face_normals[corresind, :],
            axis = 1)

            corresind = corresind[ cos_products > 0]

            if len(corresind) > 0:
                output_corres[i, :len(corresind)] = np.array(corresind)
                # print(f'{i} : {output_corres[i]}')
                # print(f'centeroid: {source_mesh.faces_centeroid[i]} , {target_mesh.faces_centeroid[output_corres[i, 0]]}')
                # print(f'normal: {source_mesh.face_normals[i]} , {target_mesh.face_normals[output_corres[i, 0]]},'
                #       f' {source_mesh.face_normals[i].dot(target_mesh.face_normals[output_corres[i, 0]])}')

        return output_corres




if __name__ == "__main__":

    ## setting
    Max_Closest_Point = 5
    dis_thershold = 0.05
    mesh_path_deformed_source = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/result/horse_camel/deformed_source/testing_more_pharse_2_3.obj'
    mesh_path_target = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/camel-poses/camel-reference.obj'
    output_corres_dir = '/Users/edwardhui/Desktop/previous_file/CSCI5210/project/result/horse_camel/corres'

    # load mesh
    ## loading meshes
    model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(mesh_path_deformed_source)
    deformed_source_mesh = Mesh(model_vertices, face_vertices_indcies)

    model_vertices, normals, face_vertices_indcies, face_normal_indice = load_mesh(mesh_path_target)
    target_mesh = Mesh(model_vertices, face_vertices_indcies)

    corres_builder = CorresBuilder(deformed_source_mesh, target_mesh)
    source_to_target_corres, target_to_source_corres = corres_builder.build_corres(Max_Closest_Point, dis_thershold)

    print(source_to_target_corres, target_to_source_corres)

    np.save(os.path.join(output_corres_dir, 'source_to_target_corres.npy'), source_to_target_corres)
    np.save(os.path.join(output_corres_dir, 'target_to_source_corres.npy'), target_to_source_corres)


