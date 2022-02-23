"""Plot the domain numbers"""
import numpy as np
import matplotlib.pyplot as plt


class PlotDomains:
    """Class for plotting local GP information"""
    def __init__(self, domain_sz, occ_map, ll, lr, ul, ur):
        """Constructor

        Args:
            domain_sz: Edge length of domain in meters
            occ_map: Occupancy grid map to represent the environment
            ll: Lower left coordinates
            ur: Upper right coordinates
            lr: Lower right coordinates
            ul: Upper left coordinates

        """
        self.domain_sz = domain_sz
        self.occ_map = occ_map
        self.ll = ll
        self.lr = lr
        self.ul = ul
        self.ur = ur

    def _get_rectangle(self, x1, x2, y1, y2):
        """Returns: Corner coordinates of a local GP"""
        return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])

    def get_rec_center(self, rect):
        """Returns: Center coordinate of a local GP"""
        p1 = rect[0]
        p2 = rect[2]
        return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])

    def plot_map(self):
        """Plot the map"""
        # Plot map
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        self.occ_map.plot(fig.axes[0], transparent=False, zorder=1)

        # Plot boundary around the entire map
        # bx1, by1 = self.occ_map.orig[:2]
        # bx2, by2 = self.occ_map.orig[:2] + np.array([self.occ_map.width, self.occ_map.height])
        # rect = self._get_rectangle(bx1, bx2, by1, by2)
        # fig.axes[0].plot(rect[:, 0], rect[:, 1], 'g', zorder=2)
        ll = self.occ_map.tf_from(np.array([0, 0]))
        lr = self.occ_map.tf_from(np.array([self.occ_map.width, 0]))
        ul = self.occ_map.tf_from(np.array([0, self.occ_map.height]))
        ur = self.occ_map.tf_from(np.array([self.occ_map.width, self.occ_map.height]))
        xmin = np.min(np.array([ll[0], lr[0], ul[0], ur[0]]))
        xmax = np.max(np.array([ll[0], lr[0], ul[0], ur[0]]))
        ymin = np.min(np.array([ll[1], lr[1], ul[1], ur[1]]))
        ymax = np.max(np.array([ll[1], lr[1], ul[1], ur[1]]))
        fig.axes[0].set_xlim(np.array([xmin, xmax]))
        fig.axes[0].set_ylim(np.array([ymin, ymax]))
        plt.pause(0.1)
        return fig

    def get_local_domain(self, number):
        """Returns: Lower left and upper right coordinate"""
        ll_rec = self.ll[number]
        lr_rec = self.lr[number]
        ul_rec = self.ul[number]
        ur_rec = self.ur[number]
        return np.array([ll_rec, ul_rec, ur_rec, lr_rec, ll_rec])

    def plot_local_domains(self):
        """Plot all local domains"""
        self.fig = self.plot_map()

        # Plot boundaries of the local domains
        for rec in range(self.ll.shape[0]):
            rect = self.get_local_domain(rec)
            self.fig.axes[0].plot(rect[:, 0], rect[:, 1], 'b', zorder=2)

            x, y = self.get_rec_center(rect)
            self.fig.axes[0].text(x, y, str(rec), color='b', fontsize=20,
                                  verticalalignment='center', horizontalalignment='center', zorder=3)
        plt.pause(0.1)

    def set_status(self, number, status):
        """To change color and information texts while optimization"""
        if status == 0:
            color = "y"
            text = "Optimize..."
        elif status == 1:
            color = "g"
            text = "Optimized"
        else:
            raise ValueError

        rect = self.get_local_domain(number)
        pos = self.get_rec_center(rect)

        # Change color of the number
        self.fig.axes[0].text(pos[0], pos[1], str(number), color=color, fontsize=20, verticalalignment='center',
                              horizontalalignment='center', zorder=4)

        # Replace information text
        pos = pos + np.array([0, -self.domain_sz / 5])
        for old_text in self.fig.axes[0].texts:
            if np.array_equal(np.array(old_text.get_position()), pos):
                old_text.remove()
        self.fig.axes[0].text(pos[0], pos[1], text, color=color, fontsize=10, verticalalignment='center',
                              horizontalalignment='center', zorder=4)
        plt.pause(0.1)


# if __name__ == "__main__":
#     # Edge length of local domains in m
#     domain_sz = 20
#     project_path = "data_io"
#
#     # Read data
#     if project_path == "data_io":
#         occ_map = mongo_utils.get_stored_occ_map()
#     elif project_path == "data":
#         r = dh.DataSim()
#         r.read_occ_map(Path(__file__).parents[1] / "data/simu_new/Map_HG.yaml")
#         occ_map = r.occ_map
#     else:
#         raise ValueError
#
#     # # Divide the map into local GP
#     # gp = PeopleGP.PeopleGP(occ_map=occ_map)
#     # gp.domain_sz = domain_sz
#     # gp.divide_map()
#
#     # Create instance
#     local_domains = PlotDomains(domain_sz=domain_sz, occ_map=occ_map, ll=gp.ll, ur=gp.ur)
#     local_domains.plot_local_domains()
#
#     # test animation:
#     # for i in range(5):
#     #     local_domains.set_status(i, 0)
#     #     plt.pause(1)
#     #     local_domains.set_status(i, 1)
#
#     plt.show()
