"""Module providing optimization class"""
import numpy as np
import gpflow
import tensorflow as tf
import time
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable
from gpflow.ci_utils import ci_niter  # not supported by gpflow 1.5.1
# import matplotlib as mpl
# mpl.use('Agg')  # No display
import matplotlib.pyplot as plt
from matplotlib import cm
from copa_map.model.model_utils import get_kernel_instance
import copa_map.util.util as ut
from tensorflow.python.framework.errors_impl import InvalidArgumentError


from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)


class Optimization:
    """Optimization class"""
    def __init__(self, method, max_duration_for_opt, max_iters_for_opt, ftol=1e-3,
                 seed=0, log_dir=None, rmse_clbl=None):
        """
        Constructor

        Args:
            method: Method for consideration of the environment structure in covariance function.
                Can be one of the following:
                bresenham: Multiplication of stationary covariance function with adjustment matrix.
                            (Iterative procedure using Bresenham algorithm)
                without_map: The environment structure will be neglected
            max_duration_for_opt: Maximum duration for optimizer in minutes
            max_iters_for_opt: Maximum iterations for optimizer
            ftol: Tolerance for difference between two consecutive loss values. If difference is below this value,
            optimization is stopped.
            seed: Seed for evaluation
            param_for_everything: For variation of any parameter during evaluation
            log_dir: Absolute path to the directory where tensorboard log files should be written to. If None, logging
                     is disabled
            rmse_clbl: Callback which calculates an error that will be plotted to tensorboard
        """
        self.log_dir = log_dir
        self.model = None
        self.model_backup = None
        self.stored_model = None
        self.method = method
        self.max_dur_opt = max_duration_for_opt
        self.max_iters_opt = max_iters_for_opt
        self.seed = seed
        self.ftol = ftol
        self.rmse_clbl = rmse_clbl

    def set_monitoring(self, model, loss_fun, xmin, xmax):
        """
        Set the monitors to log the optimization progress to tensorboard

        Args:
            model: GPFlow model to log
            loss_fun: loss function that provides the current loss during optimization
            xmin: Minimal value of input space (for image scaling)
            xmax: Maximal value for input space (for image scaling)

        """
        def plot_inducing(fig, ax):
            if hasattr(model.inducing_variable, "inducing_variables"):
                Z = model.inducing_variable.inducing_variables[0].Z
            else:
                Z = model.inducing_variable.Z
            ax.scatter(Z[:, 0], Z[:, 1], c=Z[:, 2], cmap=cm.jet)
            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(self.xmin, self.xmax)

        self.xmin = xmin
        self.xmax = xmax
        model_task = ModelToTensorBoard(self.log_dir, model, keywords_to_monitor=["kernel", "likelihood", "q_"])
        # Directory where TensorBoard files will be written.
        ut.logger().info("Saving logs in folder '" + self.log_dir + "'")
        use_inducing = hasattr(model, "inducing_variable")

        lml_task = ScalarToTensorBoard(self.log_dir, loss_fun, "loss")
        fast_tasks = MonitorTaskGroup([model_task, lml_task], period=1)
        tasks = [fast_tasks]
        if use_inducing:
            image_task = ImageToTensorBoard(self.log_dir, plot_inducing, "inducing_points",
                                            fig_kw={"figsize" : (10, 10)})
            slow_tasks = MonitorTaskGroup(image_task, period=5)
            tasks.append(slow_tasks)
        if self.rmse_clbl is not None:
            rmse_task = ScalarToTensorBoard(self.log_dir, self.rmse_clbl, "RMSE_ref")
            rmse_tasks = MonitorTaskGroup(rmse_task, period=20)
            tasks.append(rmse_tasks)

        monitor = Monitor(*tasks)
        return monitor

    def optimize(self, model, train_iter, train_dataset=None, minibatch_size=0, xmin=0, xmax=1, **kwargs):
        """Optimize model with selected optimizer.

        Args:
            model: Model to be trained
            train_iter: Training data for loss function if inducing points are used

        Returns:
            True if optimization was successful
        """
        tf.print("Optimizing...")
        self.model = model
        failed = False

        if minibatch_size > 0 and train_dataset is not None:
            train_iter = iter(train_dataset.batch(minibatch_size))

        def build_loss():
            if train_iter is None:
                _loss = self.model.training_loss_closure(compile=False)
            else:
                _loss = self.model.training_loss_closure(train_iter, compile=False)
            return _loss

        loss = build_loss()

        # GPFlow configuration
        gpflow.config.set_default_jitter(1e-3)

        self.model_backup = gpflow.utilities.deepcopy(self.model)
        time_start = time.time()
        if self.log_dir is not None:
            monitor = self.set_monitoring(model, loss, xmin, xmax)
        else:
            monitor = None
        print(f"Maximum duration for optimization of the domain: {self.max_dur_opt / 60.:.3f} minutes")
        while True:
            try:
                plt.pause(1e-6)
                self.opt_run(loss=loss, time_start=time_start, iterations=ci_niter(self.max_iters_opt),
                             monitor=monitor, **kwargs)
                break
            except KeyboardInterrupt:
                break
            except RuntimeError as e:
                print(e)
                break
            except InvalidArgumentError as e:
                print("Cancel optimization because:")
                tf.print(str(e))
                if self.method == "bresenham":
                    failed = False
                    break
                else:

                    failed = True
                    break
            except Exception as e:
                print("Cancel optimization because:")
                tf.print(str(e))
                failed = True
                break

        print("Optimized kernel parameters:")
        # Restore model with lowest loss
        gpflow.utilities.print_summary(self.model.kernel)
        return not failed

    def opt_run(self, optimizer, loss, time_start, iterations, monitor: Monitor,
                run_with_nearest_pd=False):
        """Attempt of model optimization

        Complete optimization process

        Args:
            optimizer: Optimizer to use
            loss: Loss function
            time_start: Start time to cancel optimization if required
            iterations: Maximum iterations for optimizer
            monitor: Monitor to write optimization progress to tensorboard
            run_with_nearest_pd: Set true if even after the 10th iteration step in case of instability the kernel matrix
                                 should be replaced by the nearest positive definite kernel matrix

        """
        if run_with_nearest_pd is False and self.method == "bresenham":
            print("Evaluation has shown that run_with_nearest_pd=True leads to better results. "
                  "(You can change the value inside PeopleGP._learn())")
        if optimizer == "L-BFGS-B":
            def callback(step, variables, values):
                plt.pause(1e-6)
                if monitor is not None:
                    monitor(step)
                if step % 5 == 0 or step < 10:
                    loss_new = loss()
                    print(f"step {step}: loss: {loss_new:.3f}")
                    # Check duration of optimization
                    if time.time() - time_start > self.max_dur_opt:
                        raise RuntimeError("Reached maximum optimization duration")
                # For the first 10 runs, the nearest positive definite matrix was computed if necessary when the
                # kernel was initialized accordingly. This will ensure that an optimized kernel is found in any case
                if step == 10 and self.method == "bresenham":
                    get_kernel_instance(self.model.kernel, "Mat52_Map").nearest_pd = run_with_nearest_pd
                    print(f"Set run_with_nearest_pd = {run_with_nearest_pd}")

            # SciPy optimizer: Any additional keyword arguments that are passed to the minimize method are passed
            # directly through to the SciPy optimizer’s minimize method:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            opt = gpflow.optimizers.Scipy()
            opt.minimize(closure=loss, variables=self.model.trainable_variables, compile=True, method="L-BFGS-B",
                         step_callback=callback, options=dict(disp=False, maxiter=iterations, ftol=self.ftol), )
            print(f"Training loss: {loss().numpy():.3f}")
            return self.model

        elif optimizer == "adam_natgrad" or optimizer == "adam":
            def run_adam_natgrad(model, its):
                logf = []
                # Choose callback function
                if optimizer == "adam":
                    @tf.function
                    def optimization_step():
                        ar_decremented = tf.where(tf.greater(ar, ar_min), ar - ar_step, ar_min)
                        ar.assign(ar_decremented)
                        adam_opt.minimize(loss, var_list=model.trainable_variables)
                else:
                    @tf.function
                    def optimization_step():
                        natgrad_opt.minimize(loss, var_list=variational_params)
                        gamma_decremented = tf.where(tf.greater(gamma, gamma_min), gamma - gamma_step, gamma_min)
                        gamma.assign(gamma_decremented)
                        adam_opt.minimize(loss, var_list=model.trainable_variables)
                        ar_decremented = tf.where(tf.greater(ar, ar_min), ar - ar_step, ar_min)
                        ar.assign(ar_decremented)

                for step in range(its):
                    time_start = time.time()
                    optimization_step()
                    time_end = time.time()
                    if monitor is not None:
                        monitor(step)
                    plt.pause(1e-6)
                    if step % 10 == 0 or step < 10:
                        elbo = -loss().numpy()
                        logf.append(elbo)
                        print(f"step {step}: loss: {-elbo:.3f}. Step took {(time_end-time_start):.1f} seconds.")

                        # Check on the basis of previous loss values whether the optimization can be stopped
                        # if np.var(logf[-5:])/abs(np.mean(logf[-5:])) < 1e-6 and len(logf) > 2:
                        # if np.diff(logf[-2:]) / max(np.abs(logf[-2:] + [1])) < self.ftol:

                        if step > 3 and np.abs(logf[-2] - elbo) < self.ftol:
                            print("Reached ftol")
                            break
                        # Check duration of optimization
                        if time.time() - time_start > self.max_dur_opt:
                            print("Reached maximum optimization duration")
                            break
                    if step == 10 and self.method == "bresenham":
                        get_kernel_instance(self.model.kernel, "Mat52_Map").nearest_pd = run_with_nearest_pd
                        print(f"Set run_with_nearest_pd = {run_with_nearest_pd}")
                return logf

            # Decrease the learning rate for every step
            ar_start = 1e-2
            ar_min = 1e-3
            ar_step = 3e-5
            ar = tf.Variable(ar_start, dtype=tf.float64)
            # Initialize Adam optimizer for both cases
            adam_opt = tf.optimizers.Adam(learning_rate=ar)  # epsilon=1e-3, amsgrad=True, clipnorm=1.)

            # And initialize Natural Gradient optimizer
            if optimizer == "adam_natgrad":
                # Decrease gamma every step
                gamma_start = 1e-1
                gamma_min = 1e-2
                gamma_step = 1e-3
                gamma = tf.Variable(gamma_start, dtype=tf.float64)
                # linear decrease up to the min value:

                gamma.assign(gamma_start)
                natgrad_opt = NaturalGradient(gamma=gamma)
                variational_params = [(self.model.q_mu, self.model.q_sqrt)]
                set_trainable(self.model.q_mu, False)
                set_trainable(self.model.q_sqrt, False)

            run_adam_natgrad(self.model, iterations)

        else:
            raise NotImplementedError(f'Optimization method "{optimizer}" not supported. '
                                      f'Available optimizers are "L-BFGS-B", "adam", "adam_natgrad"')
