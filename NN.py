import numpy as np
import numpy.typing as npt
import numpy.random as npr
import typing


class NN:
    _func: typing.Callable[..., npt.NDArray[np.float64]]

    def __init__(self, *layer_shape: int) -> None:
        self.w: list[npt.NDArray[np.float64]] = []
        self.b: list[npt.NDArray[np.float64]] = []
        for prev, curr in zip(layer_shape, layer_shape[1:]):
            wi: npt.NDArray[np.float64] = npr.randn(prev, curr) / np.sqrt(2 / prev)
            bi = np.zeros((1, curr))
            self.w.append(wi)
            self.b.append(bi)

    def func(self, fn: typing.Callable[..., npt.NDArray[np.float64]]) -> typing.Self:
        """set nn activation function"""
        self._func = fn
        return self

    def __call__(self, x: npt.NDArray[np.float64 | np.int64] | list[list[int | float]]):
        z: list[npt.NDArray[np.float64]] = []
        a: list[npt.NDArray[np.float64]] = []
        for wi, bi in zip(self.w, self.b):
            zi = (x @ wi) + bi
            ai = self._func(zi)
            z.append(zi)
            a.append(ai)
            x = ai
        return x, z, a


class NN_MSE(NN):
    pass


class NN_GA(NN):
    pass
