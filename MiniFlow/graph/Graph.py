from Input import Input


##
# TODO(krypten): Add documentation.
##
class Graph(object):
    @staticmethod
    def topological_sort(feed_dict=None):
        """
        Sort generic nodes in topological order using Kahn's Algorithm.

        `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

        Returns a list of sorted nodes.
        """
        if feed_dict is None:
            # Use nodes from the graph
            raise NotImplemented

        input_nodes = [n for n in feed_dict.keys()]

        G = {}
        nodes = [n for n in input_nodes]
        while len(nodes) > 0:
            n = nodes.pop(0)
            if n not in G:
                G[n] = {'in': set(), 'out': set()}
            for m in n.outbound_nodes:
                if m not in G:
                    G[m] = {'in': set(), 'out': set()}
                G[n]['out'].add(m)
                G[m]['in'].add(n)
                nodes.append(m)

        L = []
        S = set(input_nodes)
        while len(S) > 0:
            n = S.pop()

            if isinstance(n, Input):
                n.value = feed_dict[n]

            L.append(n)
            for m in n.outbound_nodes:
                G[n]['out'].remove(m)
                G[m]['in'].remove(n)
                # if no other incoming edges add to S
                if len(G[m]['in']) == 0:
                    S.add(m)
        return L

    @staticmethod
    def forward_pass(output_node, sorted_nodes):
        """
        Performs a forward pass through a list of sorted nodes.

        Arguments:

            `output_node`: The output node of the graph (no outgoing edges).
            `sorted_nodes`: a topologically sorted list of nodes.

        Returns the output node's value
        """
        if sorted_nodes is None:
            # topological_sort the graph nodes
            raise NotImplemented

        # Forward pass
        for n in sorted_nodes:
            n.forward()

    @staticmethod
    def forward_and_backward(output_node, sorted_nodes):
        """
        Performs a forward and backward pass through a list of sorted nodes.

        Arguments:

            `output_node`: The output node of the graph (no outgoing edges).
            `sorted_nodes`: a topologically sorted list of nodes.

        Returns the output node's value
        """
        Graph.forward_pass(output_node, sorted_nodes)

        # Backward pass
        for n in sorted_nodes[::-1]:
            n.backward()

    @staticmethod
    def sgd_update(trainables, learning_rate=1e-2):
        """
        Updates the value of each trainable with SGD.

        Arguments:
            `trainables`: A list of `Input` nodes representing weights/biases.
            `learning_rate`: The learning rate.
        """
        # Performs SGD
        #
        # Loop over the trainables
        for t in trainables:
            # Change the trainable's value by subtracting the learning rate
            # multiplied by the partial of the cost with respect to this
            # trainable.
            t.value -= learning_rate * t.gradients[t]

