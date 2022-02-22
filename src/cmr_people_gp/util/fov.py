"""
FOV module

Contains different field of view shapes
"""

import numpy as np
import multiprocessing as mp


class FOV:
    """
    FOV Class

    Represents a field of view and provides methods to check if positions lie
    within a field of view
    """

    def __init__(self):
        """Constructor"""
        self.position = None

    def set_position(self, pos):
        """
        Set the position of the FOV

        Args:
            pos: Position to set

        """
        self.position = pos

    def visible(self, testpos, occ_map=None, self_pos=None):
        """
        Test a position, or array of positions, for visibility.

        Args:
            testpos: A position to check (2 x 1), or an array of positions to check (2 x num of data)
            occ_map: Occupancy grid to use (optional)

        Returns:
            Boolean value, or array of size (1 x num of data)
        """
        if self_pos is not None:
            self.position = self_pos

        if self.position is None:
            raise NameError

        in_vol = self._in_volume(testpos)
        # If there is no map, or the testpoint is outside of viewing range, we do not need to raycast
        if (occ_map is None) or (type(in_vol) == np.bool_ and not in_vol):
            return in_vol
        if type(in_vol) == np.bool_:
            return occ_map.raycast(self.position, testpos, in_grid_frame=False)

        # Test positions that are in viewing range for obstacles
        visible = in_vol
        if np.sum(in_vol) > 0:
            visible[in_vol] = occ_map.raycast(self.position, testpos[in_vol], in_grid_frame=False)
        return visible

    def path_mask(self, poses, grid, occ_map=None, cpu_count=1):
        """
        Create a cell mask from available poses and grids

        Uses the the FOV from poses to determine which cells are visible. An Occupancy grid can be used to check for
        walls
        Args:
            poses: An array of poses (num x 2)
            grid: Grid to check visibility of
            occ_map: Occupancy grid map to check if view is obstructed by obstacles
            cpu_count: Number of CPU cores to use

        Returns:
            A binary mask in the same shape like grid
        """
        # Use multiprocessing
        if poses.shape[0] > cpu_count:
            indexes = np.array_split(np.arange(0, poses.shape[0]), cpu_count)
            pool = mp.Pool(cpu_count)
            result = pool.starmap(self._worker_get_cell_mask, [(poses[index], grid, occ_map) for index in indexes])
            pool.close()
            mask_fov = sum(result) > 0
        else:
            result = self._worker_get_cell_mask(poses, grid, occ_map)
            mask_fov = result

        return mask_fov

    def _worker_get_cell_mask(self, poses, grid, occ_map):
        """Worker function used in get_cell_mask for multiprocessing"""
        mask_fov = grid.empty_mask()
        # Vectorization actually not faster
        for item in range(len(poses)):
            pos = poses[item]
            mask_pos = self.visible_cells(pos=pos, grid=grid, occ_map=occ_map, mask=mask_fov)
            mask_fov = np.bitwise_or(mask_fov, mask_pos)
        return mask_fov

    def visible_cells(self, pos, grid, occ_map=None, mask=None):
        """
        Determine the visible cells

        Based on a position and a grid, checks which cells of the grid are visible from the position
        An occupancy grid map can be given to represent the environment and check if the line of sight
        is e.g. occluded by walls.

        Mask parameter can be used to speed up the computation and avoid multiple checks of the same cell

        Args:
            pos: Position to check
            grid: Grid object to use
            occ_map: Occupancy grid to use (optional)
            mask: Mask, sized like `grid`, to optionally exclude cell elements from occupancy check
                  (True values in mask will be ignored)

        Returns:
            Boolean mask, shaped like `grid`, with True values for visible cells

        """
        self.position = pos
        return self._visible_cells(grid, occ_map, mask)

    def _visible_cells(self, grid, occ_map, mask):
        raise NotImplementedError()

    def _in_volume(self, testpose):
        raise NotImplementedError()


class Circle(FOV):
    """Circle shaped field of view"""

    def __init__(self, r):
        """
        Constructor

        Args:
            r: Radius of the circle
        """
        self.r = r

    def _visible_cells(self, grid, occ_map, mask=None):
        # Get upper and lower indices of outer square
        grid_pos = grid.tf_to(self.position)
        i_u = grid.index_from_pos(grid_pos + self.r)
        i_l = grid.index_from_pos(grid_pos - self.r)
        outer_square = grid.centers_grid[i_l[0]:i_u[0], i_l[1]:i_u[1], :]
        shape_orig = outer_square.shape

        if mask is None:
            mask_small = np.full(shape_orig[0:2], True, dtype=bool)
        else:
            # Create a mask from the mask array in the same shape as the cell center vector
            mask_small = ~mask[i_l[0]:i_u[0], i_l[1]:i_u[1]]

        # When mask masks all values of square, do not check anything
        if np.all(~mask_small):
            mask_small = ~mask_small

        else:
            mask_flat = mask_small.flatten()
            # Reshape to vector
            outer_sq_vec = outer_square.reshape(-1, outer_square.shape[-1])
            # Transform back to world frame
            sq_vec_world = grid.tf_from(outer_sq_vec[mask_flat])
            # Determine which cells inside the square are visible
            mask_clip = self.visible(sq_vec_world, occ_map=occ_map)
            mask_small[mask_small] = mask_clip

        #    .reshape(shape_orig[0:2])
        # mask_clip = mask_clip.reshape(shape_orig[0:2])
        # Create mask with original grid size
        mask_ret = grid.empty_mask()
        mask_ret[i_l[0]:i_u[0], i_l[1]:i_u[1]] = mask_small
        return mask_ret

    def _in_volume(self, testpos):
        if testpos.shape == (2,):
            return (self.position[0] - testpos[0]) ** 2 + (self.position[1] - testpos[1]) ** 2 <= self.r ** 2
        else:
            return np.sum((self.position - testpos) ** 2, 1) <= self.r ** 2
