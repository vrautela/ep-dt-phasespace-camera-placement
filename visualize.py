import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def sph2cart(r, theta, phi):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z

def convert_array_to_cartesian(a):
    a2 = []
    for v in a:
        ox, oy, oz = sph2cart(1, v[3], v[4])
        a2.append([v[0], v[1], v[2], ox, oy, oz])
    
    return np.array(a2)

# TODO: change main so that it reads from result.txt
def main():
    # res = [ 0.19698639,  0.47300143,  0.1079476 ,  0.74370847,  1.01204864,
    #     0.25011296,  0.43056421,  2.13258286,  2.19820571,  0.75889565,
    #     0.1020174 ,  8.66475121,  0.01842754,  1.1588392 ,  5.29545831,
    #     0.94364443, 11.56544843,  2.73584376,  2.43000191,  5.55274855,
    #     2.93529861,  0.36033861,  0.07577052,  0.94905967,  3.00632277,
    #     3.8156321 ,  0.6361321 ,  2.7896321 ,  2.12239495,  2.85430612,
    #     3.7325057 , 11.48098716,  0.0993339 ,  1.15662046,  3.59915896,
    #     3.56099398, 11.59603136,  2.72509049,  2.28916842,  3.7784115 ]

    res = [ 0.15208953,  0.41470837,  0.14167566,  1.09330739,  0.73908804,
        0.08182775,  0.38898396,  2.80639514,  2.1260087 ,  0.8094658 ,
        0.11021482, 11.44863116,  0.09666127,  1.06809172,  5.37997749,
        0.09179937, 11.50576853,  2.75039422,  2.20732516,  5.58059941,
        3.83063347,  0.38267047,  0.08737888,  1.1591049 ,  2.44031178,
        3.89367088,  0.24102442,  2.83428936,  2.19352159,  2.51847046,
        3.84511644, 11.54322612,  0.09125082,  1.26672622,  3.77436118,
        3.89433325, 11.60313932,  2.78029694,  2.14965256,  3.74062094]

    pre_soa = [[res[5*i], res[5*i+1], res[5*i+2], res[5*i+3], res[5*i+4]] for i in range(8)]

    soa = convert_array_to_cartesian(pre_soa)

    X, Y, Z, U, V, W = zip(*soa)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 12])
    ax.set_zlim([0, 3])
    plt.show()

if __name__ == "__main__":
    main()