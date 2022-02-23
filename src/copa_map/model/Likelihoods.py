"""Module for custom specified likelihoods"""
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp


class HeteroskedasticMLGaussian(gpflow.likelihoods.QuadratureLikelihood):
    """
    Multi-Latent (Hereoscedastic) Gaussian Likelihood

    This is a multi-latent Gaussian Likelihood, where the variance parameter of the Gaussian is given by a second
    latent function.
    """

    def __init__(self, scale_transform=tfp.bijectors.Softplus(), **kwargs):
        """
        Constructor

        Args:
            scale_transform: Link function to ensure for a positive variance parameter
            **kwargs: Keyword arguments forwarded to :class:`QuadratureLikelihood`.
        """
        self.scale_transform = scale_transform
        # this likelihood expects two latent functions F, and one columns in the data vector Y:
        super().__init__(latent_dim=2, observation_dim=1, **kwargs)

    def _log_prob(self, F, Y):
        Y = tf.expand_dims(Y[:, 0], axis=-1)
        return tf.squeeze(self.conditional_distribution(F).log_prob(Y), axis=-1)

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        def conditional_y_squared(*F):
            return self._conditional_variance(*F) + tf.square(self._conditional_mean(*F))

        E_y, E_y2 = self.quadrature([self._conditional_mean, conditional_y_squared], Fmu, Fvar)
        V_y = E_y2 - E_y ** 2
        return E_y, V_y

    def conditional_distribution(self, Fs) -> tfp.distributions.Distribution:
        """
        The conditional distribution of the two-dimensional latent function F

        Args:
            Fs: Latent function where first column corresponds to latent function f, second column to g

        Returns:
            Normal distribution object
        """
        tf.debugging.assert_equal(tf.shape(Fs)[-1], 2)
        loc = Fs[..., :1]
        scale = self.scale_transform(Fs[..., 1:])
        return tfp.distributions.Normal(loc, scale)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return self.conditional_distribution(F).mean()

    def _conditional_variance(self, F):
        return self.conditional_distribution(F).variance()
