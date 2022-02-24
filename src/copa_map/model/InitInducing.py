"""Module for inducing point initialization routine"""
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from time import time

class InducingInitializer:
    """Class to create inducing point locations by clustering"""
    def __init__(self, X, Y_all, alpha=0.02, seed=None):
        """
        Constructor

        Args:
            X: [n x 3] numpy array with spatio-temporal grid centers
            Y_all: [n x 4] array with rate, counts, std. dev and observation duration of cells
            alpha: Ratio of all datapoints that determines the number of inducing points
            seed: Random seed for reproduceability
        """
        self.X = X
        self.Y_all = Y_all
        self.alpha = alpha
        self.weight = self.Y_all[:, 3]
        self.seed = seed
        self.Z = None

    def _cluster_kmeans_3d(self, num_inducing):
        """
        Cluster the spatio-temporal data completely
        Args:
            num_inducing: Number of inducing points (clusters)

        Returns:
            Cluster centers
        """
        km = KMeans(init='k-means++', n_clusters=num_inducing,
                    random_state=self.seed).fit(self.X, sample_weight=self.weight.reshape(-1))
        centers = km.cluster_centers_
        return centers

    def _cluster_kmeans_2d(self, num_inducing):
        """
        Cluster data of each timestamp separately
        Args:
            num_inducing: Number of inducing points (clusters)

        Returns:
            Cluster centers
        """
        timestamps = np.unique(self.X[:, 2])
        num_cells = self.X.shape[0]
        centers = np.array([]).reshape(0, 3)
        for t in timestamps:
            t_cond = self.X[:, 2] == t
            weight_i = self.weight[t_cond]
            X_i = self.X[t_cond]
            num_cells_ti = np.count_nonzero(t_cond)
            num_inducing_i = int(np.round(num_cells_ti / num_cells * num_inducing))
            km = KMeans(init='k-means++', n_clusters=num_inducing_i,
                        random_state=self.seed).fit(X_i, sample_weight=weight_i.reshape(-1))
            centers_i = km.cluster_centers_
            centers = np.vstack([centers, centers_i])
        return centers

    def get_init_inducing(self, method):
        """
        Create initial locations for inducing points


        Args:
            method: Can be either "2D-KMeans" or "3D-KMeans"

        Returns:
            [m x 3] array of inducing point locations
        """
        # Calculate total number of inducing points
        m = int(np.floor(self.alpha * self.X.shape[0]))
        print("Doing " + method + " for " + str(m) + " clusters")
        t_s = time()
        if method == "3D-KMeans":
            centers = self._cluster_kmeans_3d(m)
        elif method == "2D-KMeans":
            centers = self._cluster_kmeans_2d(m)
        else:
            raise NotImplementedError("Invalid inducing selection method")
        t_e = time()
        print(method + " needed " + str(t_e - t_s)  + " seconds")
        # Sort by timestamp
        centers = centers[centers[:, 2].argsort()]
        centers[:, 2] = np.round(centers[:, 2])
        self.Z = centers
        return centers

    def output_to_text(self, path):
        """
        Write the output values (X, Y, Z) to csv files.
        Args:
            path: Path, including a file prefix, where the files should be saved

        Returns:
            -
        """
        assert self.Z, "Call get_init_inducing(...) before saving"
        pdZ = pd.DataFrame(data=self.Z, columns=["x_z1", "x_z2", "t_z"])
        pdZ.to_csv(path + "_z.csv", index=False)
