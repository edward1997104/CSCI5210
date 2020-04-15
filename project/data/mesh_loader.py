import numpy as np

class Mesh(object):
    def __init__(self, vertices, face_vertices_indcies, build_triangle_features = True):
        self.vertices = vertices
        self.real_vertices_num = vertices.shape[0]
        self.face_vertices_indcies = face_vertices_indcies

        print("Starting building mesh informations......")
        # build adj relation
        self.end_vertices_to_face = self.build_end_vertices_to_face()

        # build normals
        self.build_normals()

        # build centroid
        self.build_centeroid()

        ### build triangle features
        self.origin_v_matrices = np.zeros((self.face_vertices_indcies.shape[0], 3, 3))
        self.coefficient_matrices = np.zeros((self.face_vertices_indcies.shape[0], 3, 4))

        if build_triangle_features:
            self.build_triangle_feature()

        print("Done building mesh informations......")

    def build_centeroid(self):
        self.faces_centeroid = np.zeros((self.face_vertices_indcies.shape[0], 3))

        for i in range(self.face_vertices_indcies.shape[0]):
            self.faces_centeroid[i] = np.mean(
                self.vertices[self.face_vertices_indcies[i, :3]], axis = 0
            )

    def build_normals(self):

        # face normals
        self.face_normals = np.zeros((self.face_vertices_indcies.shape[0], 3))
        for i in range(self.face_vertices_indcies.shape[0]):
            v1, v2, v3 = self.vertices[self.face_vertices_indcies[i][0]], \
                         self.vertices[self.face_vertices_indcies[i][1]], \
                         self.vertices[self.face_vertices_indcies[i][2]]
            # compute v4
            e2_1 = v2 - v1
            e3_1 = v3 - v1
            crossed_product = np.cross(e2_1, e3_1)
            length = np.sqrt(crossed_product.dot(crossed_product))
            face_normal = crossed_product / length

            self.face_normals[i] = face_normal

        # vertex normals
        self.vertices_normals = np.zeros((self.real_vertices_num, 3))
        for i in range(self.real_vertices_num):
            face_ids = set()
            for key in self.end_vertices_to_face[i].keys():
                face_ids = (face_ids | set(self.end_vertices_to_face[i][key]))

            face_normals_from_i = np.array([self.face_normals[face_id] for face_id in face_ids])

            vertex_normal = np.mean(face_normals_from_i, axis = 0)
            length = np.sqrt(vertex_normal.dot(vertex_normal))

            self.vertices_normals[i] = vertex_normal / length


    def build_end_vertices_to_face(self):
        edge_to_face_dict = [{} for i in range(self.vertices.shape[0])]
        for i in range(self.face_vertices_indcies.shape[0]):
            edge_to_face_dict = Mesh.add_triangle_to_edge_dict(edge_to_face_dict, self.face_vertices_indcies[i], i)

        return edge_to_face_dict

    def build_triangle_feature(self):

        ## compute forth vertices
        new_face_indices = []
        for i in range(self.face_vertices_indcies.shape[0]):
            v1, v2, v3 = self.vertices[self.face_vertices_indcies[i][0]], \
                         self.vertices[self.face_vertices_indcies[i][1]], \
                         self.vertices[self.face_vertices_indcies[i][2]]

            # compute v4
            e2_1 = v2 - v1
            e3_1 = v3 - v1
            crossed_product = np.cross(e2_1, e3_1)
            length = np.sqrt(crossed_product.dot(crossed_product))
            crossed_product = crossed_product / length
            v4 = v1 + crossed_product
            new_face_indices.append(self.vertices.shape[0])

            # print(f'index : {self.vertices.shape[0]}, v4 : {v4}')
            self.vertices = np.vstack((self.vertices, v4))

        self.face_vertices_indcies = np.hstack((self.face_vertices_indcies, np.array(new_face_indices)[:, np.newaxis]))

        ## compute  4 * 3 mulitple matrix for each face
        for i in range(self.face_vertices_indcies.shape[0]):
            v2_1 = self.vertices[self.face_vertices_indcies[i][1]][:, np.newaxis] - self.vertices[self.face_vertices_indcies[i][0]][:, np.newaxis]
            v3_1 = self.vertices[self.face_vertices_indcies[i][2]][:, np.newaxis] - self.vertices[self.face_vertices_indcies[i][0]][:, np.newaxis]
            v4_1 = self.vertices[self.face_vertices_indcies[i][3]][:, np.newaxis] - self.vertices[self.face_vertices_indcies[i][0]][:, np.newaxis]
            origin_v_matrix = np.hstack((v2_1,
                                         v3_1,
                                         v4_1))

            self.origin_v_matrices[i] = origin_v_matrix

            V_inv = np.linalg.inv(origin_v_matrix)
            self.coefficient_matrices[i] = np.hstack((-np.sum(V_inv, axis=0).T[:,np.newaxis], V_inv.T))



    def get_adj_face(self, face_id):
        adj_face = set()
        triangle_end_vertices = self.face_vertices_indcies[face_id]
        set_0 = set(self.end_vertices_to_face[triangle_end_vertices[0]][triangle_end_vertices[1]])
        set_1 = set(self.end_vertices_to_face[triangle_end_vertices[1]][triangle_end_vertices[0]])
        set_2 = set(self.end_vertices_to_face[triangle_end_vertices[0]][triangle_end_vertices[2]])
        set_3 = set(self.end_vertices_to_face[triangle_end_vertices[2]][triangle_end_vertices[0]])
        set_4 = set(self.end_vertices_to_face[triangle_end_vertices[1]][triangle_end_vertices[2]])
        set_5 = set(self.end_vertices_to_face[triangle_end_vertices[2]][triangle_end_vertices[1]])

        adj_face = (adj_face | set_0 | set_1 | set_2 | set_3 | set_4 | set_5) - {face_id}

        return adj_face

    @staticmethod
    def add_triangle_to_edge_dict(end_vertices_to_face_dict, triangle_tuple, face_id):
        end_vertices_to_face_dict = Mesh.add_face_indices_to_dict(end_vertices_to_face_dict, triangle_tuple[0], triangle_tuple[1],
                                                                  face_id)
        end_vertices_to_face_dict = Mesh.add_face_indices_to_dict(end_vertices_to_face_dict, triangle_tuple[1], triangle_tuple[0],
                                                                  face_id)
        end_vertices_to_face_dict = Mesh.add_face_indices_to_dict(end_vertices_to_face_dict, triangle_tuple[0], triangle_tuple[2],
                                                                  face_id)
        end_vertices_to_face_dict = Mesh.add_face_indices_to_dict(end_vertices_to_face_dict, triangle_tuple[2], triangle_tuple[0],
                                                                  face_id)
        end_vertices_to_face_dict = Mesh.add_face_indices_to_dict(end_vertices_to_face_dict, triangle_tuple[1], triangle_tuple[2],
                                                                  face_id)
        end_vertices_to_face_dict = Mesh.add_face_indices_to_dict(end_vertices_to_face_dict, triangle_tuple[2], triangle_tuple[1],
                                                                  face_id)
        return end_vertices_to_face_dict

    @staticmethod
    def add_face_indices_to_dict(end_vertices_to_face_dict, index_0, index_1, face_id):
        if index_1 not in end_vertices_to_face_dict[index_0]:
            end_vertices_to_face_dict[index_0][index_1] = [face_id]
        else:
            temp_list = end_vertices_to_face_dict[index_0][index_1]
            temp_list.append(face_id)
            end_vertices_to_face_dict[index_0][index_1] = temp_list

        return end_vertices_to_face_dict

def load_mesh(path):
    vertices, normals, face_vertices_indcies, face_normal_indices = [], [], [], []
    with open(path, mode= 'r') as f:
        for line in f.readlines():
            splited_text = line.split()
            if splited_text[0] == 'v':
                vertices.append([float(splited_text[1]), float(splited_text[2]), float(splited_text[3])])
            elif splited_text[0] == 'vn':
                normals.append([float(splited_text[1]), float(splited_text[2]), float(splited_text[3])])
            elif splited_text[0] == 'f':
                if "//" in splited_text[1]:
                    face_vertices_indcies.append([int(splited_text[1].split('//')[0])-1, int(splited_text[2].split('//')[0])-1, int(splited_text[3].split('//')[0])-1])
                    face_normal_indices.append([int(splited_text[1].split('//')[1])-1, int(splited_text[2].split('//')[1])-1, int(splited_text[3].split('//')[1])-1])
                else: # only face
                    face_vertices_indcies.append([int(splited_text[1]) - 1, int(splited_text[2]) - 1, int(splited_text[3]) - 1])

        return np.array(vertices), np.array(normals), np.array(face_vertices_indcies), np.array(face_normal_indices)


if __name__ == "__main__":
    vertices, normals, face_vertices_indcies, face_normal_indices = load_mesh('/Users/edwardhui/Desktop/previous_file/CSCI5210/project/data/horse-poses/horse-01.obj')

    mesh = Mesh(vertices, face_vertices_indcies)

    for s in range(len(mesh.end_vertices_to_face)):
        for d in mesh.end_vertices_to_face[s].keys():
            if len(mesh.end_vertices_to_face[s][d]) == 2:
                pass
            else:
                cnt = 0
                for i in range(mesh.face_vertices_indcies.shape[0]):
                    face_indices_list = face_vertices_indcies[i].tolist()
                    if s in face_indices_list and d in face_indices_list:
                        cnt += 1

                assert cnt == len(mesh.end_vertices_to_face[s][d])

    for i in range(mesh.face_vertices_indcies.shape[0]):
        assert normals[face_normal_indices[i][0]].dot(mesh.vertices_normals[mesh.face_vertices_indcies[i][0]]) > 0