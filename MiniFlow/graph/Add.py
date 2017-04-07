from Node import Node


##
# Add node that adds the values coming from the inbound.
##
class Add(Node):
    def __init__(self, *inputs):
        # An Add node has list of inputs,
        # so we need to pass them as list to the Node instantiator.
        Node.__init__(self, inputs)

    def forward(self, value=None):
        """
        Forward propagation.

        Add all the values in from inbound_nodes and store
        the calculated value in self.value.
        """
        self.value = 0
        for node in self.inbound_nodes:
            self.value += node.value
