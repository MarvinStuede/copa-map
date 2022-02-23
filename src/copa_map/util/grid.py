"""
Module for grid

Contains a general grid class
"""
import numpy as np
import transformations as tf
from itertools import product
from copy import copy


class Grid:
    """
    General representation of a grid

    Represented by a height, width, resolution and pose of coordinate frame
    """

    def __init__(self, width, height, resolution=1, origin=None, rotation=None):
        """
        Constructor

        Args:
            width: Width of the grid (in meters) in x direction
            height: Height of the grid (in meters) in y direction
            resolution: Resolution of the grid (meters/cell)
            origin: Translational origin of the grid in world frame. Origin is the lower left corner of the map, and is
                    the origin of the grid frame, where x points right and y points up
            rotation: Rotation to grid frame from world frame
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.elements_x = int(self.width / self.resolution)
        self.elements_y = int(self.height / self.resolution)

        if origin is None:
            self.orig = [0.0, 0.0, 0.0]
        else:
            self.orig = origin

        if rotation is None:
            rot = tf.quaternion_matrix([1, 0, 0, 0])
            self.rotation = 0
        else:
            rot = tf.quaternion_matrix(tf.quaternion_about_axis(rotation, [0, 0, 1]))
            self.rotation = rotation

        self.T_hom = tf.concatenate_matrices(tf.translation_matrix(self.orig), rot)
        self.T_hom_I = np.linalg.inv(self.T_hom)

        self.centers = self._get_centers()
        self.edges = self._get_edges()
        self.centers_grid = self.centers.reshape(self.elements_x, self.elements_y, 2)

    def tf_from(self, pos):
        """
        Transform a position from grid frame to world frame

        Args:
            pos:  A position to transform

        Returns: Position in world frame

        """
        return self._tf(pos, self.T_hom)

    def tf_to(self, pos):
        """
        Transform a position, or array of positions,  from world frame to grid frame

        Args:
            pos: A position to transform (2,)
                 or an array of positions (Data points x 2)

        Returns: Position(s) in grid frame

        """
        return self._tf(pos, self.T_hom_I)

    def _tf(self, pos, T):
        # Convert a matrix
        if pos.ndim == 2 and pos.shape[1] == 2:
            # zero for third coordinate, homogenize
            poses_hom = np.hstack((pos, np.zeros((pos.shape[0], 1)), np.ones((pos.shape[0], 1))))
            # Apply dot product for every column
            result = np.sum(T * poses_hom[:, None], 2)
            return result[:, :2]
        # Convert a vector
        else:
            assert pos.shape == (2,)
            pe = np.append(pos, [0, 1])
            p_t = np.dot(T, pe)
            return p_t[0:2]

    def index_from_pos(self, pos, clip=False):
        """
        Convert a position in grid frame to array index

        Args:
            pos: Position to convert
            clip: Clip index to valid range

        Returns: Array index

        """
        if pos.ndim == 2 and pos.shape[1] == 2:
            e_x = np.unique(self.edges[:, 0])
            e_y = np.unique(self.edges[:, 1])
            ix = np.digitize(pos[:, 0], e_x) - 1
            iy = np.digitize(pos[:, 1], e_y) - 1
            ix = np.clip(ix, 0, self.elements_x - 1)
            iy = np.clip(iy, 0, self.elements_y - 1)
            ind = np.column_stack([ix, iy])
        else:
            ix = np.clip(int(np.floor(pos[0] / self.resolution)), 0, self.elements_x - 1)
            iy = np.clip(int(np.floor(pos[1] / self.resolution)), 0, self.elements_y - 1)
            ind = np.array([ix, iy])
        assert np.all(ix < self.elements_x), "At least one pos outside of grid"
        assert np.all(iy < self.elements_y), "At least one pos outside of grid"
        return ind

    def pos_from_index(self, index):
        """
        Convert an index to a position in grid frame

        Args:
            index: Index to convert

        Returns:
            Position in grid frame
        """
        pos_y = index[0] * self.resolution
        pos_x = index[1] * self.resolution

        return pos_x, pos_y

    def is_index_valid(self, index):
        """
        Check if an index is a valid index of the grid

        Args:
            index: index to check

        Returns:
            True if valid
        """
        vx = 0 <= index[0] <= (self.elements_x - 1)
        vy = 0 <= index[1] <= (self.elements_y - 1)
        return (vx and vy)

    def get_centers(self, as_3d=False, in_grid_frame=False):
        """
        Get the centers of the grid cells

        Args:
            as_3d: Return centers with first two dimensions sized like the grid
            in_grid_frame: Return centers in grid frame, else in world frame

        Returns: Centers of the grid cells as coordinates

        """
        if in_grid_frame:
            bin = self.centers
        else:
            bin = np.apply_along_axis(self.tf_from, 1, self.centers)

        if as_3d:
            return bin.reshape(self.elements_x, self.elements_y, 2)
        else:
            return bin

    def get_edges(self, as_3d=False, in_grid_frame=False):
        """
        Get the edges of the grid cells

        Args:
            as_3d: Return edges with first two dimensions sized like the grid
            in_grid_frame: Return edges in grid frame, else in world frame

        Returns: Edges of the grid cells as coordinates

        """
        if in_grid_frame:
            bin = self.edges
        else:
            bin = np.apply_along_axis(self.tf_from, 1, self.edges)

        if as_3d:
            return bin.reshape(self.elements_x + 1, self.elements_y + 1, 2)
        else:
            return bin

    def pos_in_grid(self, pos, in_grid_frame=False, extend=0):
        """
        Check which poses are on the grid

        Args:
            pos: [2xn] array of positions to check
            in_grid_frame: If False, positions will be transformed to grid frame
            extend: Check extended area of grid extended by this amount in meters

        Returns:
            Boolean array of length n indicating which poses are on the grid
        """
        if in_grid_frame:
            pos_check = copy(pos)
        else:
            pos_check = self.tf_to(copy(pos))

        return (pos_check[:, 0] > -extend) & (pos_check[:, 1] > -extend) & (pos_check[:, 0] <= self.width + extend) & \
               (pos_check[:, 1] <= self.height + extend)

    def _get_centers(self):
        """Returns: Centers of the grid cells as coordinates"""
        lin_x = np.linspace(0, self.width, self.elements_x + 1)
        lin_y = np.linspace(0, self.height, self.elements_y + 1)
        offset_x = lin_x[:-1] + 0.5 * np.diff(lin_x)[0]
        offset_y = lin_y[:-1] + 0.5 * np.diff(lin_y)[0]
        bin_mids = np.array([np.array(elem) for elem in product(offset_x, offset_y)])

        return bin_mids

    def _get_edges(self):
        """Returns: Edges of the grid cells as coordinates"""
        lin_x = np.linspace(0, self.width, self.elements_x + 1)
        lin_y = np.linspace(0, self.height, self.elements_y + 1)
        bin_edges = np.array([np.array(elem) for elem in product(lin_x, lin_y)])

        return bin_edges

    def empty_mask(self):
        """Returns: Grid sized mask with all false values"""
        return np.full((self.elements_x, self.elements_y), False, dtype=bool)
