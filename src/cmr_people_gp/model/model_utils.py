"""Utilities for optimization of a model"""
import gpflow
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from termcolor import colored


def get_kernel_instance(kern, name):
    """
    Returns requested kernel instance of a combined kernel

    Args:
        kern: Combined kernel
        name: Instance name

    Returns:
        kernel instance
    """

    def search_submodules(kern_, name_):
        for sub_module in kern_.kernels:
            if sub_module.name == name_:
                return sub_module
            elif sub_module.name in ["sum", "product"]:
                result = search_submodules(sub_module, name_)
                if result is not None:
                    return result
        return None

    instance = search_submodules(kern, name)
    if instance is None:
        print(colored("Kernel instance \"{}\" was not found", "red").format(name))
    return instance


def bounded_hyperparameter(low, high, parameter):
    """
    To constrain hyper-parameters

    Args:
        low: Minimum value
        high: Maximum value
        parameter: Initial value

    Returns:
        Bounded hyper-parameter
    """
    # Compute Y = g(X; shift, scale) = scale * X + shift.
    affine = tfb.AffineScalar(shift=tf.cast(low, tf.float64),
                              scale=tf.cast(high - low, tf.float64))
    # Transformation-bijector combining affine and sigmoid
    logistic = tfb.Chain([affine, tfb.Sigmoid()])
    # Apply the transformation to the parameter
    param = gpflow.Parameter(parameter, transform=logistic, dtype=tf.float64)
    return param
