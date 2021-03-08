Installation
============

Installing PyTorchSpiking
-------------------------
We recommend using ``pip`` to install PyTorchSpiking:

.. code-block:: bash

  pip install pytorch-spiking

That's it!

Requirements
------------
PyTorchSpiking works with Python 3.6 or later.  ``pip`` will do its best to install
all of PyTorchSpiking's requirements automatically.  However, if anything
goes wrong during this process you can install the requirements manually and
then try to ``pip install pytorch-spiking`` again.

Developer installation
----------------------
If you want to modify PyTorchSpiking, or get the very latest updates, you will need to
perform a developer installation:

.. code-block:: bash

  git clone https://github.com/nengo/pytorch-spiking.git
  pip install -e ./pytorch-spiking

Installing PyTorch
---------------------
The PyTorch documentation has a
`useful tool <https://pytorch.org/get-started/locally/#start-locally>`_ to determine
the appropriate command to install pytorch on your system. We would recommend using
``conda``, as it will take care of installing the other GPU-related packages, like
CUDA/cuDNN.
