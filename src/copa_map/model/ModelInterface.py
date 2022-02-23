"""General module for model with parameters"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelParams:
    """Basic parameters for People model"""
    # Seed for evaluation
    seed: int = None


class ModelInterface(ABC):
    """Abstract class to interact with a PeopleGP classlike object"""

    @abstractmethod
    def learn(self, X, Y, Z, **kwargs):
        """
        Abstract learn method

        Args:
            X: Input locations
            Y: Observations corresponding to input locations
            Z: Inducing points
            **kwargs: Additional arguments for specific learn methods
        """
        pass

    @abstractmethod
    def predict(self, Xt):
        """
        Abstract predict method

        Args:
            Xt: Inputs for prediction

        Returns: mean and variance

        """
        pass

    @abstractmethod
    def get_model_params(self, **kwargs):
        """
        Return the model parameters

        Returns:
            Instance of the ModelParams class or subclass
        """
        pass

    @abstractmethod
    def write_model_params(self, params, **kwargs):
        """
        Write the model parameters

        Args:
            params: Instance of the ModelParams class or subclass

        """
        pass

    def replace_zeros_with_eps(self, pred, eps=1e-5):
        """
        Replace zero predictions with very small values

        This is necessary for some evaluations, e.g. cost paths
        Args:
            pred: Model predictive output
            eps: Small value to set

        Returns:
            Model predicted with replaced values
        """
        pred[pred == 0.] = eps
        return pred
