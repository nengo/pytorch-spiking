"""Functional implementation of spiking layers."""

import torch


class SpikingActivation(torch.autograd.Function):
    """
    Function for converting an arbitrary activation function to a spiking equivalent.

    Notes
    -----
    We would not recommend calling this directly, use
    `pytorch_spiking.SpikingActivation` instead.
    """

    @staticmethod
    def forward(
        ctx,
        inputs,
        activation,
        dt=0.001,
        initial_state=None,
        spiking_aware_training=True,
        return_sequences=False,
        training=False,
    ):
        """
        Forward pass of SpikingActivation function.

        Parameters
        ----------
        inputs : ``torch.Tensor``
            Array of input values with shape ``(batch_size, n_steps, n_neurons)``.
        activation : callable
            Activation function to be converted to spiking equivalent.
        dt : float
            Length of time (in seconds) represented by one time step.
        initial_state : ``torch.Tensor``
            Initial spiking voltage state (should be an array with shape
            ``(batch_size, n_neurons)``, with values between 0 and 1). Will use a
            uniform distribution if none is specified.
        spiking_aware_training : bool
            If True (default), use the spiking activation function
            for the forward pass and the base activation function for the backward pass.
            If False, use the base activation function for the forward and
            backward pass during training.
        return_sequences : bool
            Whether to return the last output in the output sequence (default), or the
            full sequence.
        training : bool
            Whether this function should be executed in training or evaluation mode
            (this only matters if ``spiking_aware_training=False``).
        """

        ctx.activation = activation
        ctx.return_sequences = return_sequences
        ctx.save_for_backward(inputs)

        if training and not spiking_aware_training:
            output = activation(inputs if return_sequences else inputs[:, -1])
            return output

        if initial_state is None:
            initial_state = torch.rand(
                inputs.shape[0], inputs.shape[2], dtype=inputs.dtype
            )

        # match inputs to initial state dtype if one was passed in
        inputs = inputs.type(initial_state.dtype)

        voltage = initial_state
        all_spikes = []
        rates = activation(inputs) * dt
        for i in range(inputs.shape[1]):
            voltage += rates[:, i]
            n_spikes = torch.floor(voltage)
            voltage -= n_spikes
            if return_sequences:
                all_spikes.append(n_spikes)

        if return_sequences:
            output = torch.stack(all_spikes, dim=1)
        else:
            output = n_spikes

        output /= dt

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of SpikingActivation function."""

        # TODO: is there a way to reuse the forward pass activations computed in
        #  `forward`? the below results in an infinite loop
        # inputs, rates = ctx.saved_tensors
        # return torch.autograd.grad(rates, inputs, grad_outputs=grad_output)

        inputs = ctx.saved_tensors[0]
        with torch.enable_grad():
            output = ctx.activation(inputs if ctx.return_sequences else inputs[:, -1])
            return (
                torch.autograd.grad(output, inputs, grad_outputs=grad_output)
                + (None,) * 7
            )


spiking_activation = SpikingActivation.apply
