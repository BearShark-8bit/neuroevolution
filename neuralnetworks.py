import numpy
from typing import Callable

type listOfFloats = numpy.typing.NDArray[numpy.float64]
type matrixOfFloats = list[listOfFloats]


class NeuralNetwork:
    activationFunction: Callable[..., listOfFloats]

    def __init__(
        self,
        *networkShape: int,
        **config: Callable[..., listOfFloats],
    ) -> None:
        self.weights: matrixOfFloats = []
        self.biases: matrixOfFloats = []
        try:
            self.activationFunction = config["activationFunction"]
        except KeyError:
            self.activationFunction = lambda x: x
        for prev, curr in zip(networkShape, networkShape[1:]):
            wi: listOfFloats = numpy.random.randn(prev, curr) / numpy.sqrt(2 / prev)
            bi = numpy.zeros((1, curr))
            self.weights.append(wi)
            self.biases.append(bi)

    def __call__(self, x: listOfFloats):
        z: matrixOfFloats = []
        a: matrixOfFloats = []
        for wi, bi in zip(self.weights, self.biases):
            zi = (x @ wi) + bi
            ai = self.activationFunction(zi)
            z.append(zi)
            a.append(ai)
            x = ai
        return x, z, a


# class NN_MSE(NeuralNetwork):
#     def __init__(
#         self,
#         *networkShape: int,
#         **config: Callable[..., listOfFloats],
#     ) -> None:
#         super().__init__(*networkShape, **config)
#         try:
#             self.activationFunction_prime = config["activationFunction_prime"]
#         except KeyError:
#             self.activationFunction_prime = lambda x: x**0

#     def learn(
#         self,
#         x: listOfFloats,
#         y: listOfFloats,
#         gamma: float,
#     ) -> numpy.float64:
#         """train and return loss MSE"""
#         pred, z, a = self(x)
#         chain = 1
#         for i in range(len(self.weights) - 1, -1, -1):
#             print(i)

#         return numpy.mean((pred - y) ** 2).astype(numpy.float64)


class EvolvingNeuralNetwork(NeuralNetwork):
    def __init__(
        self,
        *networkShape: int,
        **config: Callable[..., listOfFloats],
    ) -> None:
        super().__init__(*networkShape, **config)

    def mutate(self):
        pass

    def __add__(self, other: "EvolvingNeuralNetwork"):  # crossover
        pass
