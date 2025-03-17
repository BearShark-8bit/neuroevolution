import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt
import numpy.random as npr
import typing


class NN:
    func: typing.Callable[..., npt.NDArray[np.float64]]

    def __init__(
        self, *layer_shape: int, **config: typing.Callable[..., npt.NDArray[np.float64]]
    ) -> None:
        self.w:  list[npt.NDArray[np.float64]] = []
        self.b: list[npt.NDArray[np.float64]] = []
        try:
            self.func = config["func"]
        except KeyError:
            self.func = lambda x: x
        for prev, curr in zip(layer_shape, layer_shape[1:]):
            wi: npt.NDArray[np.float64] = npr.randn(prev, curr) / np.sqrt(2 / prev)
            bi = np.zeros((1, curr))
            self.w.append(wi)
            self.b.append(bi)

    def __call__(self, x: npt.NDArray[np.float64]):
        z: list[npt.NDArray[np.float64]] = []
        a: list[npt.NDArray[np.float64]] = []
        for wi, bi in zip(self.w, self.b):
            zi = (x @ wi) + bi
            ai = self.func(zi)
            z.append(zi)
            a.append(ai)
            x = ai
        return x, z, a


class NN_MSE(NN):
    def __init__(
        self, *layer_shape: int, **config: typing.Callable[..., npt.NDArray[np.float64]]
    ) -> None:
        super().__init__(*layer_shape, **config)
        try:
            self.func_prime = config["func_prime"]
        except KeyError:
            self.func_prime = lambda x: x**0

    def learn(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        gamma: float,
    ) -> np.float64:
        """train and return loss MSE"""
        pred, z, a = self(x)
        chain = 1
        for i in range(len(self.w)-1, -1, -1):
            print(i)

        return np.mean((pred - y) ** 2).astype(np.float64)


class NN_GA(NN):
    pass
