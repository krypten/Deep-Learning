import numpy as np
from Node import Node


##
# Applying the sigmoid activation function to the input.
##
class Sigmoid(Node):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """

    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used later with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.
        """
        return 1. / (1. + np.exp(-x))

    def _sigmoid_prime(self, x):
        return self._sigmoid(x) * (1. - self._sigmoid(x))

    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.
        """
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the cost with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += self._sigmoid_prime(self.inbound_nodes[0].value) * grad_cost
