"""Modules for adding spiking behaviour to PyTorch models."""

import numpy as np
import torch

from pytorch_spiking.functional import spiking_activation


class SpikingActivation(torch.nn.Module):  # pylint: disable=abstract-method
    """Module for converting an arbitrary activation function to a spiking equivalent.

    Neurons will spike at a rate proportional to the output of the base activation
    function. For example, if the activation function is outputting a value of 10, then
    the wrapped SpikingActivationCell will output spikes at a rate of 10Hz (i.e., 10
    spikes per 1 simulated second, where 1 simulated second is equivalent to ``1/dt``
    time steps). Each spike will have height ``1/dt`` (so that the integral of the
    spiking output will be the same as the integral of the base activation output).
    Note that if the base activation is outputting a negative value then the spikes
    will have height ``-1/dt``. Multiple spikes per timestep are also possible, in
    which case the output will be ``n/dt`` (where ``n`` is the number of spikes).

    When applying this layer to an input, make sure that the input has a time axis.
    The spiking output will be computed along the time axis.
    The number of simulation timesteps will depend on the length of that time axis.
    The number of timesteps does not need to be the same during
    training/evaluation/inference. In particular, it may be more efficient
    to use one timestep during training and multiple timesteps during inference
    (often with ``spiking_aware_training=False``, and ``apply_during_training=False``
    on any `.Lowpass` layers).

    Parameters
    ----------
    activation : callable
        Activation function to be converted to spiking equivalent.
    dt : float
        Length of time (in seconds) represented by one time step.
    initial_state : ``torch.Tensor``
        Initial spiking voltage state (should be an array with shape
        ``(batch_size, n_neurons)``, with values between 0 and 1). Will use a uniform
        distribution if none is specified.
    spiking_aware_training : bool
        If True (default), use the spiking activation function
        for the forward pass and the base activation function for the backward pass.
        If False, use the base activation function for the forward and
        backward pass during training.
    return_sequences : bool
        Whether to return the full sequence of output spikes (default),
        or just the spikes on the last timestep.
    """

    def __init__(
        self,
        activation,
        dt=0.001,
        initial_state=None,
        spiking_aware_training=True,
        return_sequences=True,
    ):
        """"""  # empty docstring removes useless parent docstring from docs
        super().__init__()

        self.activation = activation
        self.initial_state = initial_state
        self.dt = dt
        self.spiking_aware_training = spiking_aware_training
        self.return_sequences = return_sequences

    def forward(self, inputs):
        """Compute output spikes given inputs.

        Parameters
        ----------
        inputs : ``torch.Tensor``
            Array of input values with shape ``(batch_size, n_steps, n_neurons)``.

        Returns
        -------
        outputs : ``torch.Tensor``
            Array of output spikes with shape ``(batch_size, n_neurons)`` if
            ``return_sequences=False`` else ``(batch_size, n_steps, n_neurons)``. Each
            element will have value ``n/dt``, where ``n`` is the number of spikes
            emitted by that neuron on that time step.
        """
        return spiking_activation(
            inputs,
            self.activation,
            self.dt,
            self.initial_state,
            self.spiking_aware_training,
            self.return_sequences,
            self.training,
        )


class Lowpass(torch.nn.Module):  # pylint: disable=abstract-method
    """Module implementing a Lowpass filter.

    The initial filter state and filter time constants are both trainable
    parameters. However, if ``apply_during_training=False`` then the parameters are
    not part of the training loop, and so will never be updated.

    When applying this layer to an input, make sure that the input has a time axis.

    Parameters
    ----------
    tau : float
        Time constant of filter (in seconds).
    dt : float
        Length of time (in seconds) represented by one time step.
    apply_during_training : bool
        If False, this layer will effectively be ignored during training (this
        often makes sense in concert with the swappable training behaviour in, e.g.,
        `.SpikingActivation`, since if the activations are not spiking during training
        then we often don't need to filter them either).
    level_initializer : ``torch.Tensor``
        Initializer for filter state.
    return_sequences : bool
        Whether to return the full sequence of filtered output (default),
        or just the output on the last timestep.
    """

    def __init__(
        self,
        tau,
        units,
        dt=0.001,
        apply_during_training=True,
        initial_level=None,
        return_sequences=True,
    ):
        """"""  # empty docstring removes useless parent docstring from docs
        super().__init__()

        if tau <= 0:
            raise ValueError("tau must be a positive number")

        self.tau = tau
        self.units = units
        self.dt = dt
        self.apply_during_training = apply_during_training
        self.initial_level = initial_level
        self.return_sequences = return_sequences

        # apply ZOH discretization
        smoothing_init = np.exp(-self.dt / self.tau)

        # compute inverse sigmoid of tau, so that when we apply the sigmoid
        # later we'll get the tau value specified
        self.smoothing_init = np.log(smoothing_init / (1 - smoothing_init))

        self.level_var = torch.nn.Parameter(
            torch.zeros(1, units) if self.initial_level is None else self.initial_level
        )

        self.smoothing_var = torch.nn.Parameter(
            torch.ones(1, units) * self.smoothing_init
        )

    def forward(self, inputs):
        """Apply filter to inputs.

        Parameters
        ----------
        inputs : ``torch.Tensor``
            Array of input values with shape ``(batch_size, n_steps, units)``.

        Returns
        -------
        outputs : ``torch.Tensor``
            Array of output spikes with shape ``(batch_size, units)`` if
            ``return_sequences=False`` else ``(batch_size, n_steps, units)``.
        """

        if self.training and not self.apply_during_training:
            return inputs if self.return_sequences else inputs[:, -1]

        level = self.level_var
        smoothing = torch.sigmoid(self.smoothing_var)

        # cast inputs to module type
        inputs = inputs.type(self.smoothing_var.dtype)

        all_levels = []
        for i in range(inputs.shape[1]):
            level = (1 - smoothing) * inputs[:, i] + smoothing * level
            if self.return_sequences:
                all_levels.append(level)

        if self.return_sequences:
            return torch.stack(all_levels, dim=1)
        else:
            return level


class TemporalAvgPool(torch.nn.Module):
    """Module for taking the average across one dimension of a tensor.

    Parameters
    ----------
    dim : int, optional
        The dimension to average across. Defaults to the second dimension (``dim=1``),
        which is typically the time dimension (for tensors that have a time dimension).
    """

    def __init__(self, dim=1):
        """"""  # empty docstring removes useless parent docstring from docs
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        """Apply average pooling to inputs.

        Parameters
        ----------
        inputs : ``torch.Tensor``
            Array of input values with shape ``(batch_size, n_steps, ...)``.

        Returns
        -------
        outputs : ``torch.Tensor``
            Array of output values with shape ``(batch_size, ...)``.
            The time dimension is fully averaged and removed.
        """
        return torch.mean(inputs, dim=self.dim)
