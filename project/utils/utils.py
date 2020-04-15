from data.mesh_loader import Mesh


def output_obj(path, mesh : Mesh):
    f = open(path, 'w')
    for i in range(mesh.real_vertices_num):
        f.write(f'v {mesh.vertices[i,0]} {mesh.vertices[i,1]} {mesh.vertices[i,2]}\n')

    for i in range(mesh.face_vertices_indcies.shape[0]):
        f.write(f'f {mesh.face_vertices_indcies[i][0]+1} {mesh.face_vertices_indcies[i][1]+1} {mesh.face_vertices_indcies[i][2]+1}\n')