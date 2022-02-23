"""Module for Bresenham kernel"""

import numpy as np
from cmr_people_gp.util.occ_grid import OccGrid
import cv2


class KernelGrid(OccGrid):
    """Class for creating an occupation map with widened walls"""
    def __init__(self, base_occ_map: OccGrid, digitize_size=0.2, num_of_borders=2):
        """
        Constructor

        Args:
             base_occ_map: Occupancy grid map to use as basis of the kernel. The Kernel grid will have the same
                           dimension and origin as the map
             digitize_size: Discretization size for grid bins
             num_of_borders: Number of cells around occupied cells, from which covariance factor increases linearly
             from 0 to 1
        """
        # We do not need the full map resolution, so we resize the image based on the given parameter
        assert digitize_size >= base_occ_map.resolution,\
            "Kernel map discretization should be larger than Occupancy grid map resolution"

        # Rescale the occupancy map
        new_img_size = (np.array(base_occ_map.img.shape) * base_occ_map.resolution / digitize_size).astype(int)
        new_img = cv2.resize(base_occ_map.img, dsize=(new_img_size[1], new_img_size[0]),
                             interpolation=cv2.INTER_NEAREST_EXACT)

        super(KernelGrid, self).__init__(img=new_img,
                                         width=base_occ_map.width,
                                         height=base_occ_map.height,
                                         resolution=digitize_size,
                                         origin=base_occ_map.orig,
                                         rotation=base_occ_map.rotation,
                                         )
        self.digitize_size = digitize_size
        self.num_of_borders = num_of_borders
        self._create_map()

    def _create_map(self):
        """
        Creates a grid array characterizing walls and cells near walls

        Reads the map and creates cells with the defined digitize_size, where walls are classified with 0
        and free cells with 1. The values of surrounding cells increase linearly to 1 depending on the
        number of neighboring cells num_of_borders
        """
        # Create kernel for dilation. Every pixels 8-neighbors should be extended
        kernel = np.ones((3, 3), np.uint8)
        # Get factor between extension border which determines the occupancy
        # Interpolates linearly so that every border increases occupancy by same amount
        increment = 1 / (self.num_of_borders + 1)
        adj_img = dil_img = self.img
        # Extend the wall pixels by dilating the image, then multiplying with the respective factor for occupancy
        # reduction
        for i in np.arange(0, 1, increment):
            if i == 0:
                continue
            # Dilate the image from last iteration by one more border
            # Our map has zeros where we want to extend, so we need to use the inverse
            dil_img = cv2.dilate(~dil_img, kernel)
            dil_img = ~dil_img
            # Change the pixels of the new border, where the old image was still white (255) and the new
            # is now black (0)
            adj_img[np.logical_and(dil_img == 0, adj_img == 255)] = i * 255

        self.img = adj_img
        self.map = np.flipud(adj_img.astype(float) / 255)
