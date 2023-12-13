import torch
import pandas as pd
import numpy as np


def get_flatten_grad(model):
    """Return the flattened parameter of a model,
    returns a (n,1) tensor with the total number of parameters"""

    parameters = list(model.parameters())
    grads = [
        param.grad.flatten().view(-1, 1)
        for param in parameters
        if not type(param.grad) == type(None)
    ]
    grad = torch.cat(grads)
    return grad


# Linear combination for the moving average
def lin_comb(v1, v2, beta):
    return beta * v1 + (1 - beta) * v2


def mom3(avg, beta, yi, i):
    "Exponential moving average with debiasing"
    if avg is None:
        avg = 0
    avg = lin_comb(avg, yi, beta)
    return avg, avg / (1 - beta ** (i + 1))


class BSFinder:
    def __init__(self, bs=None, num_it: int = None, n_batch=5, beta=0.99):
        self.b_small = bs
        self.b_big = bs * n_batch

        self.num_it = num_it
        self.n_batch = n_batch
        self.beta = beta
        self.running_scale = None
        self.running_noise = None

        self.batches = []

        # We create the list which will store the data
        self.output = []

    def on_backward_end(self, model, iteration: int, **kwargs) -> None:
        if iteration >= self.num_it:
            return {"stop_epoch": True, "stop_training": True}

        # First we grab the gradient
        grad = get_flatten_grad(model)
        self.batches.append(grad)

        if iteration % self.n_batch == self.n_batch - 1:
            # We concatenate the batches and empty the buffer

            batches = torch.cat(self.batches, dim=1)
            self.batches = []

            grads = batches.mean(dim=1)

            g_big = (grads**2).mean()
            g_small = (grad**2).mean()

            noise = (self.b_big * g_big - self.b_small * g_small) / (
                self.b_big - self.b_small
            )
            scale = (g_small - g_big) / ((1 / self.b_small) - (1 / self.b_big))

            self.running_scale, scale = mom3(
                self.running_scale, self.beta, scale, iteration
            )
            self.running_noise, noise = mom3(
                self.running_noise, self.beta, noise, iteration
            )

            scale = scale.item()
            noise = noise.item()
            noise_scale = scale / noise

            self.output.append(
                {"noise": noise, "scale": scale, "noise_scale": noise_scale}
            )

    def on_train_end(self, **kwargs) -> None:
        "Cleanup learn model weights disturbed during exploration."
        pass

    def plot(self):
        "Plot the average noise scale"
        df = pd.DataFrame(self.output)
        df.noise_scale.plot(title=f"Average Noise scale : {df.noise_scale.mean()}")
