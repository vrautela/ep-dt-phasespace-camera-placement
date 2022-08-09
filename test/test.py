from optimal_camera_placement.optimization.optimizer import gen_guess_box, gen_guess_cube
from optimal_camera_placement.optimization.optimizer import in_fov, objective_function
import optimal_camera_placement.optimization.optimizer as optimizer
import numpy as np


def test_in_fov():
    halfway = optimizer.fov_upper_bound / 2
    pos_vec = np.array([0, 0, 0])
    orientation_vec = np.array([0, 1, 0])

    grid_point = np.array([0, halfway, halfway]) 
    assert(in_fov(pos_vec, orientation_vec, grid_point))
    grid_point = np.array([halfway, halfway, 0]) 
    assert(in_fov(pos_vec, orientation_vec, grid_point))

    grid_point = np.array([0, halfway, -halfway]) 
    assert(in_fov(pos_vec, orientation_vec, grid_point))
    grid_point = np.array([-halfway, halfway, 0]) 
    assert(in_fov(pos_vec, orientation_vec, grid_point))

    grid_point = np.array([0, optimizer.fov_upper_bound - 1, 0]) 
    assert(in_fov(pos_vec, orientation_vec, grid_point))
    

def test_not_in_fov():
    halfway = optimizer.fov_upper_bound / 2
    pos_vec = np.array([0, 0, 0])
    orientation_vec = np.array([0, 1, 0])

    grid_point = np.array([0,-1,0])
    assert(not in_fov(pos_vec, orientation_vec, grid_point))

    grid_point = np.array([0, optimizer.fov_upper_bound + 1, 0])
    assert(not in_fov(pos_vec, orientation_vec, grid_point))

    grid_point = np.array([0, halfway, halfway * 2])
    assert(not in_fov(pos_vec, orientation_vec, grid_point))

    grid_point = np.array([0, halfway * 2, halfway * 4])
    assert(not in_fov(pos_vec, orientation_vec, grid_point))


# How to compare these two methods? They don't actually produce the same result
# Cube doesn't point towards the center
def test_gen_guess_box_vs_cube():
    assert(gen_guess_cube(5) == gen_guess_box(5,5,5))


if __name__ == "__main__":
    test_in_fov()
    test_not_in_fov()
    test_gen_guess_box_vs_cube()