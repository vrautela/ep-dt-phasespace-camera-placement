import numpy as np
from typing import Tuple

class Obstacle:
   """Abstract class representing an obstacle in a volume"""

   # Does the line segment defined by points p1 and p2 intersect this obstacle?
   def does_line_segment_intersect(self, p1: np.ndarray, p2: np.ndarray) -> bool:
      raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement does_line_segment_intersect()")


# class representing a cylindrical obstacle
class CylinderObstacle(Obstacle):
   """Class representing a cylindrical obstacle"""

   def __init__(self, A, B, R):
      # A and B are the centers of the bases of the cylinder
      # R is the radius of the cylinder
      # h is the height of the cylinder
      self.A = A
      self.B = B
      self.R = R
      self.h = np.linalg.norm(B - A) 

   def does_line_segment_intersect(self, p1: np.ndarray, p2: np.ndarray) -> bool:
      # 1. Compute P_i - A

      shifted_p1 = p1 - self.A
      shifted_p2 = p2 - self.A

      # 2. Transform shifted points into new basis (formed by rotating z-axis onto a)

      # unit vector along axis of cylinder
      a = self.B - self.A 
      a = a / np.linalg.norm(a)

      z_hat = np.array([0,0,1])
      # rotation matrix that rotates z-axis onto a
      R = self.rotation_matrix_align_two_vecs(z_hat, a)

      # Compute inverse of R to express all vecs in new basis
      R_inv = np.linalg.inv(R)

      # transform shifted points using R_inv
      new_shifted_p1 = np.matmul(R_inv, shifted_p1)
      new_shifted_p2 = np.matmul(R_inv, shifted_p2)

      # 3. Use (x', y') coords of new shifted points to see if there is
      #    an intersection of the points with a circle of radius R 
      #    centered at the origin

      (x1, y1) = (new_shifted_p1[0], new_shifted_p1[1]) 
      (x2, y2) = (new_shifted_p2[0], new_shifted_p2[1]) 

      # find (x', y') coords of intersection points using method from wolfram
      dx = x2 - x1
      dy = y2 - y1
      dr = np.sqrt(dx**2 + dy**2)
      D = x1*y2 - x2*y1

      discriminant = (self.R**2)*(dr**2) - D**2

      # negative discriminant --> no points of intersection
      # zero discriminant --> one point of intersection (tangent) 
      # TODO: check if it is okay to regard tangency as invisibility
      if discriminant <= 0:
         return False
      
      # compute (x', y') coords of intersection points
      def sgn(v):
         if v < 0:
            return -1
         else:
            return 1

      # TODO: see if both intersection points (and subsequent t values) are required
      x_int_1 = (D*dy + sgn(dy)*dx*discriminant)/(dr**2)
      y_int_1 = (-D*dx + abs(dy)*discriminant)/(dr**2) 

      x_int_2 = (D*dy - sgn(dy)*dx*discriminant)/(dr**2)
      y_int_2 = (-D*dx - abs(dy)*discriminant)/(dr**2) 

      # 4. Determine z' component of intersection points by using 2d vector form of line

      # unit vector from p2 to p1
      v = p2 - p1
      v = v / np.linalg.norm(v)
      # components of v in new basis
      new_v = np.matmul(R_inv, v)

      # using intersection point coords and parametric form 
      # of line (initial' + tv'), compute t for each intersection point
      # TODO: check t values are the same if you use the y components instead
      t1 = (x_int_1 - new_shifted_p1[0])/new_v[0]
      t2 = (x_int_2 - new_shifted_p2[0])/new_v[0]
      
      # from t values, compute the z' coords of the intersection points 
      z_int_1 = t1*new_v[2] + new_shifted_p1[2]
      z_int_2 = t2*new_v[2] + new_shifted_p1[2]

      # 5. Check if z component is between 0 and height
      #    If so, then there is an intersection between 
      #    the line segment and the cylinder
      
      return 0 < z_int_1 < self.h or 0 < z_int_2 < self.h


   def rotation_matrix_align_two_vecs(a: np.ndarray, b: np.ndarray):
      """
      R = I + [v]_x + (1/1+c)([v]_x)^2 
      """
      v = np.cross(a, b)
      c = np.dot(a, b)

      v1, v2, v3 = v[0], v[1], v[2]
      v_ss_cp = np.array([0,-v3,v2], [v3,0,-v1], [-v2,v1,0])
      
      I = np.eye(3)
      R = I + v_ss_cp + (1/(1+c))*np.linalg.matrix_power(v_ss_cp, 2)

      return R
