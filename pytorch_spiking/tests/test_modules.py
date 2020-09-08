import copy

import numpy as np
import pytest
import torch


from pytorch_spiking import modules


@pytest.mark.parametrize(
    "activation", (torch.nn.ReLU(), torch.nn.functional.relu, torch.tanh)
)
def test_activations(activation, rng, allclose):
    x = torch.from_numpy(rng.randn(32, 10, 2))

    ground = activation(x)

    # behaviour equivalent to base activation during training
    y = modules.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=False
    ).train()(x)
    assert allclose(y, ground)

    # not equivalent during inference
    y = modules.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=False
    ).eval()(x)
    assert not allclose(y, ground, record_rmse=False, print_fail=0)

    # equivalent during inference, with large enough dt
    y = modules.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=False, dt=1e8
    ).eval()(x)
    assert allclose(y, ground)

    # not equivalent during training if using spiking_aware_training
    y = modules.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=True
    ).train()(x)
    assert not allclose(y, ground, record_rmse=False, print_fail=0)

    # equivalent with large enough dt
    y = modules.SpikingActivation(
        activation,
        return_sequences=True,
        spiking_aware_training=True,
        dt=1e8,
    ).train()(x)
    assert allclose(y, ground)


def test_initial_state(seed, allclose):
    x = torch.from_numpy(np.ones((2, 100, 10)) * 100)

    init = torch.rand((2, 10), generator=torch.random.manual_seed(seed))

    # layers with the same initial state produce the same output
    y0 = modules.SpikingActivation(
        torch.nn.ReLU(), return_sequences=True, initial_state=init
    )(x)
    y1 = modules.SpikingActivation(
        torch.nn.ReLU(), return_sequences=True, initial_state=init
    )(x)
    assert allclose(y0, y1)

    # layers with different initial state produce different output
    y2 = modules.SpikingActivation(torch.nn.ReLU(), return_sequences=True)(x)
    assert not allclose(y0, y2, record_rmse=False, print_fail=0)

    # the same layer called multiple times will produce the same output (if the initial
    # state is set)
    layer = modules.SpikingActivation(
        torch.nn.ReLU(), return_sequences=True, initial_state=init
    )
    assert allclose(layer(x), layer(x))

    # layer will produce different output each time if initial state not set
    layer = modules.SpikingActivation(torch.nn.ReLU(), return_sequences=True)
    assert not allclose(layer(x), layer(x), record_rmse=False, print_fail=0)


def test_spiking_aware_training(rng, allclose):
    layer = modules.SpikingActivation(
        torch.nn.ReLU(), spiking_aware_training=False
    ).train()
    layer_sat = modules.SpikingActivation(
        torch.nn.ReLU(), spiking_aware_training=True
    ).train()
    x = torch.from_numpy(rng.uniform(-1, 1, size=(10, 20, 32))).requires_grad_(True)
    y = layer(x)[:, -1]
    y_sat = layer_sat(x)[:, -1]
    y_ground = torch.nn.ReLU()(x)[:, -1]

    # forward pass is different
    assert allclose(y.detach().numpy(), y_ground.detach().numpy())
    assert not allclose(
        y_sat.detach().numpy(),
        y_ground.detach().numpy(),
        record_rmse=False,
        print_fail=0,
    )

    # gradients are the same
    dy = torch.autograd.grad(y, x, grad_outputs=[torch.ones_like(y)])[0]
    dy_ground = torch.autograd.grad(y_ground, x, grad_outputs=[torch.ones_like(y)])[0]
    dy_sat = torch.autograd.grad(y_sat, x, grad_outputs=[torch.ones_like(y)])[0]
    assert allclose(dy, dy_ground)
    assert allclose(dy_sat, dy_ground)


def test_spiking_swap_functional(allclose):
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.dense0 = torch.nn.Linear(1, 10)
            self.dense1 = torch.nn.Linear(1, 10)

        def forward(self, inputs):
            x = inputs.view((-1, inputs.shape[-1]))

            x0 = self.dense0(x)
            x0 = x0.view((inputs.shape[0], inputs.shape[1], x0.shape[-1]))
            x0 = torch.nn.LeakyReLU(negative_slope=0.3)(x0)

            x1 = self.dense1(x)
            x1 = x1.view((inputs.shape[0], inputs.shape[1], x1.shape[-1]))
            x1 = modules.SpikingActivation(
                torch.nn.LeakyReLU(negative_slope=0.3),
                return_sequences=True,
                spiking_aware_training=False,
            )(x1)

            return x0, x1

    model = MyModel()
    loss_func = torch.nn.MSELoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for _ in range(200):
        model.zero_grad()

        outputs = model(torch.ones((32, 1, 1)))
        loss = sum(
            torch.mean(torch.sum(loss_func(o, t), dim=1))
            for o, t in zip(
                outputs,
                [torch.ones((32, 1, 10)) * torch.arange(1.0, 100.0, 10)] * 2,
            )
        )
        loss.backward()
        optimizer.step()

    y0, y1 = model(torch.ones((1, 1000, 1)))
    assert allclose(y0.detach().numpy(), np.arange(1, 100, 10), atol=1)
    assert allclose(
        np.sum(y1.detach().numpy() * 0.001, axis=1, keepdims=True),
        np.arange(1, 100, 10),
        atol=1,
    )


@pytest.mark.parametrize("dt", (0.001, 1))
def test_lowpass_tau(dt, allclose, rng):
    nengo = pytest.importorskip("nengo")

    # verify that the pytorch-spiking lowpass implementation matches the nengo lowpass
    # implementation
    layer = modules.Lowpass(tau=0.1, units=32, dt=dt).double()

    with torch.no_grad():
        x = torch.from_numpy(rng.randn(10, 100, 32))
        y = layer(x)

    y_nengo = nengo.Lowpass(0.1).filt(x, axis=1, dt=dt)

    assert allclose(y, y_nengo)


def test_lowpass_apply_during_training(allclose, rng):
    with torch.no_grad():
        x = torch.from_numpy(rng.randn(10, 100, 32))

        # apply_during_training=False:
        #   confirm `output == input` for training=True, but not training=False
        layer = modules.Lowpass(
            tau=0.1, units=32, apply_during_training=False, return_sequences=True
        )
        assert allclose(layer.train()(x), x)
        assert not allclose(layer.eval()(x), x, record_rmse=False, print_fail=0)

        # apply_during_training=True:
        #   confirm `output != input` for both values of `training`, and
        #   output is equal for both values of `training`
        layer = modules.Lowpass(
            tau=0.1, units=32, apply_during_training=True, return_sequences=True
        )
        assert not allclose(layer.train()(x), x, record_rmse=False, print_fail=0)
        assert not allclose(layer.eval()(x), x, record_rmse=False, print_fail=0)
        assert allclose(layer.train()(x), layer.eval()(x))


def test_lowpass_trainable(allclose):
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.trained = modules.Lowpass(0.01, 1, apply_during_training=True)
            self.skip = modules.Lowpass(0.01, 1, apply_during_training=False)
            self.untrained = modules.Lowpass(0.01, 1, apply_during_training=True)
            for param in self.untrained.parameters():
                param.requires_grad = False

        def forward(self, inputs):
            return self.trained(inputs), self.skip(inputs), self.untrained(inputs)

    model = MyModel()

    loss_func = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(10):
        model.zero_grad()

        outputs = model(torch.zeros(1, 1, 1))
        loss = sum(
            loss_func(o, t)
            for o, t in zip(
                outputs,
                [torch.ones(1, 1)] * 3,
            )
        )
        loss.backward()
        optimizer.step()

    # trainable layer should learn to output 1
    ys = model(torch.zeros((1, 1, 1)))
    assert allclose(ys[0].detach(), 1)
    assert not allclose(ys[1].detach(), 1, record_rmse=False, print_fail=0)
    assert not allclose(ys[2].detach(), 1, record_rmse=False, print_fail=0)

    # for trainable layer, smoothing * initial_level should go to 1
    assert allclose(
        (torch.sigmoid(model.trained.smoothing_var) * model.trained.level_var).detach(),
        1,
    )

    # other layers should stay at initial value
    assert allclose(model.skip.level_var.detach(), 0)
    assert allclose(model.untrained.level_var.detach(), 0)
    assert allclose(model.skip.smoothing_var.detach(), model.skip.smoothing_init)
    assert allclose(
        model.untrained.smoothing_var.detach(), model.untrained.smoothing_init
    )


def test_lowpass_validation():
    with pytest.raises(ValueError, match="tau must be a positive number"):
        modules.Lowpass(tau=0, units=1)


def test_temporalavgpool(rng, allclose):
    x = rng.randn(32, 10, 2, 5)
    tx = torch.from_numpy(x)
    for dim in range(x.ndim):
        model = torch.nn.Sequential(modules.TemporalAvgPool(dim=dim))
        toutput = model(tx)
        assert allclose(toutput.numpy(), x.mean(axis=dim))


@pytest.mark.parametrize(
    "module",
    (
        modules.SpikingActivation(
            torch.nn.ReLU(), initial_state=torch.zeros((32, 50)), dt=1
        ),
        modules.Lowpass(tau=0.01, units=50, dt=0.001),
    ),
)
def test_return_sequences(module, rng, allclose):
    x = torch.tensor(rng.randn(32, 10, 50))

    with torch.no_grad():
        module_seq = copy.deepcopy(module)
        module_seq.return_sequences = True
        y_seq = module_seq(x)

        module_last = copy.deepcopy(module)
        module_last.return_sequences = False
        y_last = module_last(x)

    assert y_seq.shape == x.shape
    assert y_last.shape == (x.shape[0], x.shape[2])
    assert allclose(y_seq[:, -1], y_last)
