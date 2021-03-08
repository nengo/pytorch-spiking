PyTorchSpiking versus NengoDL
=============================

If you are interested in combining spiking neurons and deep learning methods, you may
be familiar with `NengoDL <https://www.nengo.ai/nengo-dl>`_ (and wondering what the
difference is between PyTorchSpiking and NengoDL).

The short answer is that PyTorchSpiking is designed to be a lightweight, minimal
implementation of spiking behaviour that integrates very transparently into PyTorch.
It is designed to get you up and running on building a spiking model with very little
overhead.

NengoDL provides much more robust, fully-featured tools for building spiking models.
More neuron types, more synapse types, more complex network architectures, more of
everything basically. However, all of those extra features require a more significant
departure from the PyTorch API. There is more of a learning curve to
getting started with NengoDL, and because NengoDL is based on TensorFlow/Keras, the
API is designed to be more familiar to those with Keras experience.

One particularly significant distinction is that PyTorchSpiking does not really
integrate with the rest of the Nengo ecosystem (e.g., it cannot run models built with
the Nengo API, and models built with PyTorchSpiking cannot run on other Nengo
platforms).
In contrast, NengoDL can run any Nengo model, and models optimized in NengoDL can
be run on other Nengo platforms (such as custom neuromorphic hardware, like NengoLoihi).

In summary, you should use PyTorchSpiking if you want to get up and running with minimal
departures from the standard PyTorch API. If you find yourself wishing for more control
or more features to build your model, or you would like to run your model on different
hardware platforms, consider checking out NengoDL.
