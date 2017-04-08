##
# Base class to be extended by each node used for creating graph.
##

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # Keys are the inputs to this node and their values are
        # the partials of this node with respect to that input.
        self.gradients = {}
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.

        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplemented
