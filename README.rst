.. image:: https://img.shields.io/travis/com/nengo/pytorch-spiking/master.svg
  :target: https://travis-ci.com/nengo/pytorch-spiking
  :alt: Travis-CI build status

.. image:: https://img.shields.io/codecov/c/github/nengo/pytorch-spiking/master.svg
  :target: https://codecov.io/gh/nengo/pytorch-spiking
  :alt: Test coverage

**************
PyTorchSpiking
**************

PyTorchSpiking provides tools for training and running spiking neural networks
directly within the PyTorch framework. The main feature is
``pytorch_spiking.SpikingActivation``, which can be used to transform
any activation function into a spiking equivalent. For example, we can translate a
non-spiking model, such as

.. code-block:: python

    torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.ReLU(),
    )

into the spiking equivalent:

.. code-block:: python

    torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        pytorch_spiking.SpikingActivation(torch.nn.ReLU()),
    )

Models with SpikingActivation layers can be optimized and evaluated in the same way as
any other PyTorch model. They will automatically take advantage of PyTorchSpiking's
"spiking aware training": using the spiking activations on the forward pass and the
non-spiking (differentiable) activation function on the backwards pass.

PyTorchSpiking also includes various tools to assist in the training of spiking models,
such as `filtering layers
<https://www.nengo.ai/pytorch-spiking/reference.html#module-pytorch_spiking.modules>`_.

If you are interested in building and optimizing spiking neuron models, you may also
be interested in `NengoDL <https://www.nengo.ai/nengo-dl>`_. See
`this page <https://www.nengo.ai/pytorch-spiking/nengo-dl-comparison.html>`_ for a
comparison of the different use cases supported by these two packages.

**Documentation**

Check out the `documentation <https://www.nengo.ai/pytorch-spiking/>`_ for

- `Installation instructions
  <https://www.nengo.ai/pytorch-spiking/installation.html>`_
- `More detailed example introducing the features of PyTorchSpiking
  <https://www.nengo.ai/pytorch-spiking/examples/spiking-fashion-mnist.html>`_
- `API reference <https://www.nengo.ai/pytorch-spiking/reference.html>`_
