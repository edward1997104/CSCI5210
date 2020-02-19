import numpy as np

def write_point_cloud(point_arr : np.array, file_path):
    f = open(file_path, 'w+')

    for i in range(point_arr.shape[0]):
        f.write(f"v {point_arr[i][0]} {point_arr[i][1]} {point_arr[i][2]}\n")

