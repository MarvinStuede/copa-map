"""
Module for Occupancy grid

Contains a class dervied from grid to represent an occupancy map
"""

import cv2
import numpy as np
from copa_map.util import util
from matplotlib import transforms as mtf
import yaml
from copa_map.util.grid import Grid
from os.path import join, isfile, split
import imageio


class OccGrid(Grid):
    """
    Occupancy Grid class

    Represents an Occupancy Grid map. Map is represented as an np array with black values (0) for walls
    and white values (255) for free areas
    """

    def __init__(self, img, morphological_transformation=False, kernel_size=2, *args, **kwargs):
        """
        Constructor

        Creates the grid map, based on a binary image (np uint8 array)

        Args:
            img: Map as image
            morphological_transformation: Modifies map using opencv for use in kernel
            kernel_size: Size of the kernel of the morphological transformation
            *args: args of base class
            **kwargs: kwargs of base class
        """
        super(OccGrid, self).__init__(*args, **kwargs)
        if morphological_transformation:
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
            _, img = cv2.threshold(img, img.max() - 1, 255, cv2.THRESH_BINARY)
            img = util.remove_small_objects_in_image(cv2.bitwise_not(img), min_pixel_size=3 / self.resolution)
            img = cv2.bitwise_not(img)
        self.img = img
        if img is not None:
            self.map = np.flipud(self.img)

    def plot(self, plt, transparent=False, black_thresh=255, invert=False, **kwargs):  # pragma: no cover
        """
        Plots the map

        Args:
            invert: (Optional) Invert the colors of the map
            black_thresh: (Optional) Threshold over which areas will be made transparent
            transparent: If true, only black areas are plotted
            plt: Matplotlib object
        """
        extent = np.array(self.get_img_extend())

        if transparent:
            img = np.ma.masked_where(self.img >= black_thresh, self.img)
        else:
            img = self.img

        tr = mtf.Affine2D().rotate_around(self.orig[0], self.orig[1], self.rotation)
        if invert:
            cmap = "binary"
        else:
            cmap = "gist_gray"

        plt.imshow(img, extent=extent, cmap=cmap, transform=tr + plt.transData, **kwargs)

    def get_walls(self, ):
        """Determines the coordinates of wall pixels"""
        # Determines wall coordinates from occupied obstacle pixels
        walls_x, walls_y = np.where(self.map == 0)
        walls = np.column_stack((walls_y, walls_x)) * self.resolution

        # Function tf_from() does not work here. With a rotation of -90 degrees (simulation data set) the result is
        # only correct if walls is modified as follows:
        # walls = self.occ_map.tf_from(walls)
        # walls[:, 1] = walls[:, 1] + self.occ_map.height

        # Workaround: Manual Transformation
        walls = walls + self.orig[:2]  # translation
        center = (self.orig[:2] + np.array([self.width / 2, self.height / 2])).reshape(1, 2)
        R = np.array([[np.cos(self.rotation), -np.sin(self.rotation)],
                      [np.sin(self.rotation), np.cos(self.rotation)]])
        walls = (R @ (walls.T - center.T) + center.T).T  # rotation
        return walls

    def get_img_extend(self):
        """Returns: Extend of the map as a list (in meters)."""
        # Get lower left corner from transformation matrix
        m_t_min = self.T_hom[0:2, 3]
        m_t_max = m_t_min + np.array([self.width, self.height])
        # ll = self.tf_from(np.array([0, 0]))
        # lr = self.tf_from(np.array([self.width, 0]))
        # ul = self.tf_from(np.array([0, self.height]))
        # ur = self.tf_from(np.array([self.width, self.height]))
        # xmin = np.min(np.array([ll[0], lr[0], ul[0], ur[0]]))
        # xmax = np.max(np.array([ll[0], lr[0], ul[0], ur[0]]))
        # ymin = np.min(np.array([ll[1], lr[1], ul[1], ur[1]]))
        # ymax = np.max(np.array([ll[1], lr[1], ul[1], ur[1]]))
        return [m_t_min[0],
                m_t_max[0],
                m_t_min[1],
                m_t_max[1]]

    @classmethod
    def from_ros_format(cls, path_yaml):
        """
        Create an instance of this class based on a ROS map

        Standard .yaml/.png definition.
        See Also: http://wiki.ros.org/map_server
        Args:
            path_yaml: Path to the yaml file

        Returns:
            Class instance
        """
        assert isfile(path_yaml), "Path must point to a yaml file"
        try:
            with open(path_yaml) as y_file:
                fyaml = yaml.load(y_file, Loader=yaml.FullLoader)
                origin = [fyaml['origin'][0], fyaml['origin'][1], 0]
                resolution = fyaml['resolution']
                # Get read image from name in yaml
                tp1, tp2 = split(path_yaml)
                ipath = join(tp1, fyaml['image'])
                img = imageio.imread(ipath, pilmode="RGB")
                (height, width, _) = img.shape
                img = img[:, :, 0]
                inst = cls(width=width * resolution, height=height * resolution,
                           resolution=fyaml['resolution'], origin=origin,
                           rotation=fyaml['origin'][2], img=img,
                           morphological_transformation=False)
                return inst

        except OSError:
            raise OSError("Map file could not be read")

    def to_ros_format(self, path, name, fformat="png"):
        """
        Write the occupancy map to ROS compatible format

        Standard .yaml/.png definition.
        See Also: http://wiki.ros.org/map_server
        Args:
            path: Folder where the map should be written
            name: name of the map
            fformat: image format, "png" or "pgm"
        """
        # Create meta data as yaml
        yaml_dict = {
            "image": name + "." + fformat,
            "resolution": self.resolution,
            "origin": self.orig,
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,

        }

        with open(join(path, name + ".yaml"), 'w') as yaml_file:
            yaml.dump(yaml_dict, yaml_file, default_flow_style=False)

        imageio.imwrite(join(path, name + "." + format), self.img, format=format)

    def raycast(self, pos, testpos, in_grid_frame=False):
        """
        Raycast to check if line between points is occluded by the map.

        Checks start point with a single, or an array of testpoints.

        Args:
            pos: Start point (1 x dimension)
            testpos: An end point (1 x dimension)
                     or An array of end points
                     (number of points x dimension)
            in_grid_frame: Points given in frame of the grid

        Returns:
            Result as an array of (number of points x 1)

        """
        return self._generic_raycast_dim_check(pos, testpos, in_grid_frame, self._ray_unoccluded)

    def raycast_to(self, pos, testpos, in_grid_frame=False):
        """
        Raycast from a position to one or multiple test pos, and return first collision

        Args:
            pos: Start pos of ray (1 x dimension)
            testpos: An end point (1 x dimension)
                     or An array of end points
                     (number of points x dimension)
            in_grid_frame: Points given in frame of the grid

        Returns:
            Coordinates where first collision with map obstacle occurs. If no collision, coordinate of
            end position is returned
        """
        return self._generic_raycast_dim_check(pos, testpos, in_grid_frame, self._ray_to)

    def _generic_raycast_dim_check(self, pos, testpos, in_grid_frame, raycast_fun):
        if testpos.shape == (2,):
            return raycast_fun(pos, testpos, in_grid_frame)
        else:
            return np.apply_along_axis(func1d=raycast_fun, axis=1,
                                       arr=testpos, p2=pos, in_grid_frame=in_grid_frame)

    def _ray_to(self, p1, p2, in_grid_frame=False):
        if in_grid_frame:
            p1i = self.index_from_pos(p1)
            p2i = self.index_from_pos(p2)
        else:
            p1i = self.index_from_pos(self.tf_to(p1))
            p2i = self.index_from_pos(self.tf_to(p2))

        coord = util.min_on_line_coord(p2i, p1i, self.map.T, thresh=50)
        pos = np.flipud(np.array(self.pos_from_index(coord)))
        if in_grid_frame:
            return pos
        else:
            return self.tf_from(pos)

    def _ray_unoccluded(self, p1, p2, in_grid_frame=False):
        if in_grid_frame:
            p1i = self.index_from_pos(p1)
            p2i = self.index_from_pos(p2)
        else:
            p1i = self.index_from_pos(self.tf_to(p1))
            p2i = self.index_from_pos(self.tf_to(p2))

        # Create a line of indices between the points
        list_i = util.bresenhamline(np.asarray([p1i]), np.asarray([p2i]))

        # Get the values of the underlying map
        occ_ray = self.map[tuple(map(tuple, np.fliplr(list_i).T))]
        occluded = np.any(a=occ_ray == 0)

        return ~occluded
