import numpy as np
from Node import Node


##
# Linear solution for the node according to summation where x coming from the inbound.
##
class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        """
        Forward propagation.

        Calculate value in self.value
        """
        if self.inbound_nodes is not None:
            inputs = self.inbound_nodes[0].value
            weights = self.inbound_nodes[1].value
            bias = self.inbound_nodes[2].value
            self.value = np.dot(inputs, weights) + bias
