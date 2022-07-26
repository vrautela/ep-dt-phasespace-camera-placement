import matplotlib.pyplot as plt
import numpy as np
from optimizer import gdop
# Test of GDOP vs triangulability metrics


# create a fixed camera, a fixed point, and a camera that moves around the point in a circle
def two_cam_test():
    c1 = [3, 0, 0, 90, 180] 
    p = (0, 0, 0)

    x = []
    y = []

    # make c2 move in a circle of radius 3 around the origin
    for alpha in range(360):
        # define c2 to be a camera which is alpha degrees along the circumference of a circle of radius 3
        # phi can be from [0, 360]
        phi = (alpha + 180) % 360
        c2 = [3 * np.cos(np.deg2rad(alpha)), 3 * np.sin(np.deg2rad(alpha)), 0, 90, phi]

        cams = []
        cams.extend([c1[0], c1[1], c1[2]]) 
        cams.extend([c2[0], c2[1], c2[2]]) 

        x.append(alpha)
        y.append(gdop(cams, p))


    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('Angle between cams (degrees)')
    ax.set_ylabel('GDOP')
    plt.show()


# set up two fixed cams in the xy plane (with angular separation 135 degrees) and have a third cam
# rotate in the yz plane
def three_cam_test():
    c1 = [3, 0, 0, 90, 180] 
    c3 = [3 * np.cos(np.deg2rad(135)), 3 * np.sin(np.deg2rad(135)), 0, 90, 315]
    p = (0, 0, 0)

    x = []
    y = []

    # make c2 move in a circle of radius 3 around the origin in the yz plane
    for alpha in range(360):
        # define c2 to be a camera which is alpha degrees along the circumference of a circle of radius 3
        # phi can be from [0, 360]
        phi = (alpha + 180) % 360
        c2 = [0, 3 * np.cos(np.deg2rad(alpha)), 3 * np.sin(np.deg2rad(alpha)), 90, phi]

        cams = []
        cams.extend([c1[0], c1[1], c1[2]]) 
        cams.extend([c2[0], c2[1], c2[2]]) 
        cams.extend([c3[0], c3[1], c3[2]]) 

        x.append(alpha)
        y.append(gdop(cams, p))


    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('Angle between cam and xy plane (degrees)')
    ax.set_ylabel('GDOP')
    plt.show()


def main():
    three_cam_test()

if __name__ == '__main__':
    main()