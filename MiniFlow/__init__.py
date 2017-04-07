from graph.Input import Input
from graph.Add import Add
from graph.Linear import Linear
from graph.Sigmoid import Sigmoid
from graph.Graph import Graph

import numpy as np

X, W, b = Input(), Input(), Input()
print('------- Sigmod Output ----------')
f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = Graph.topological_sort(feed_dict)
output = Graph.forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
print (output)

print('------------------------------')
