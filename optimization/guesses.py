import math
import numpy as np

# Initial guess for 8-camera system with V=5x5x5 (after shifting variables off the boundary)
x0 = [
    0.1,0.1,0.1,np.deg2rad(45),np.deg2rad(45),
    0.1,0.1,4.9,np.deg2rad(45),np.deg2rad(135),
    0.1,4.9,0.1,np.deg2rad(135),np.deg2rad(45),
    0.1,4.9,4.9,np.deg2rad(135),np.deg2rad(135),
    4.9,0.1,0.1,np.deg2rad(45),np.deg2rad(315),
    4.9,0.1,4.9,np.deg2rad(45),np.deg2rad(225),
    4.9,4.9,0.1,np.deg2rad(135),np.deg2rad(315),
    4.9,4.9,4.9,np.deg2rad(135),np.deg2rad(225),
]
# Optimal solution found with above guess
# with 2 point objective function
'''
[
    1.55359568, 0.        , 0.        , 0.53960293, 1.57079633,
    1.45398252, 0.        , 0.        , 1.56080525, 1.59600433,
    0.00623887, 4.88892499, 0.1       , 2.35619449, 0.78539816,
    0.1       , 4.9       , 4.9       , 2.35619449, 2.35619449,
    4.9       , 0.1       , 0.1       , 0.78539816, 5.49778714,
    4.9       , 0.1       , 4.9       , 0.78539816, 3.92699082,
    4.9       , 4.9       , 0.1       , 2.35619449, 5.49778714,
    4.9       , 4.9       , 4.9       , 2.35619449, 3.92699082
]
'''


def gen_guess_cube(s: int):
    t = s - 0.1
    return [
        0.1, 0.1, 0.1, np.deg2rad(45), np.deg2rad(45),
        0.1, 0.1, t, np.deg2rad(135), np.deg2rad(45),
        0.1, t, 0.1, np.deg2rad(45), np.deg2rad(315),
        0.1, t, t, np.deg2rad(135), np.deg2rad(315),
        t, 0.1, 0.1, np.deg2rad(45), np.deg2rad(135),
        t, 0.1, t, np.deg2rad(135), np.deg2rad(135),
        t, t, 0.1, np.deg2rad(45), np.deg2rad(225),
        t, t, t, np.deg2rad(135), np.deg2rad(225),
    ]


def gen_guess_box(l, w, h):
    x0 = []

    center_x = l/2
    center_y = w/2
    center_z = h/2

    center = np.array([center_x, center_y, center_z])

    margin = 0.05
    delta_l = l * margin
    delta_w = w * margin
    delta_h = h * margin

    c1 = np.array([0,0,0])
    c2 = np.array([0,0,h])
    c3 = np.array([0,w,0])
    c4 = np.array([0,w,h])
    c5 = np.array([l,0,0])
    c6 = np.array([l,0,h])
    c7 = np.array([l,w,0])
    c8 = np.array([l,w,h])
    cams = [c1, c2, c3, c4, c5, c6, c7, c8]

    for cam in cams:
        orientation_vec = center - cam
        o_x = orientation_vec[0]
        o_y = orientation_vec[1]
        o_z = orientation_vec[2]

        theta = np.arccos(o_z / math.sqrt(o_x**2 + o_y**2 + o_z**2))
        phi = None
        if o_x > 0:
            phi = np.arctan(o_y / o_x)
        elif o_x < 0 and o_y > 0:
            phi = np.arctan(o_y / o_x) + math.pi
        elif o_x < 0 and o_y < 0:
            phi = np.arctan(o_y / o_x) - math.pi
        elif x == 0 and y > 0:
            phi = math.pi / 2
        elif x == 0 and y < 0:
            phi = -math.pi / 2
        elif x == 0 and y == 0:
            raise Exception("x and y cannot both be 0 to convert into spherical coordinates")
        assert(phi is not None)

        # ensure that all angles are between 0 and 360 degrees
        if phi < 0:
            phi += (2 * math.pi)
        
        cam_x = cam[0]
        cam_y = cam[1]
        cam_z = cam[2]

        x = delta_l if cam_x == 0 else l - delta_l 
        y = delta_w if cam_y == 0 else w - delta_w
        z = delta_h if cam_z == 0 else h - delta_h
        x0.extend([x, y, z, theta, phi])

    return x0