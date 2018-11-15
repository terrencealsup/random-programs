import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



# Given a probability transition matrix P,
# construct the associated digraph.
# P is nxn matrix where n is the number
# of states.

def make_graph(P):

    G = nx.DiGraph()
    G.add_nodes_from(range(len(P)))
    for i in range(len(P)):
        for j in range(len(P)):
            if P[i][j] > 0:
                G.add_edges_from([(i,j)], weight=P[i][j])
    return G


# P_1 = np.array([[1./3, 0, 2./3, 0, 0, 0],
#                 [0, 1./4, 0, 3./4, 0, 0],
#                 [2./3, 0, 1./3, 0, 0, 0],
#                 [0, 1./5, 0, 4./5, 0, 0],
#                 [1./4, 1./4, 0, 0, 1./4, 1./4],
#                 [1./6, 1./6, 1./6, 1./6, 1./6, 1./6]])


P_1 = np.array([[1,0,0,0,0,0],
               [0.5,0,0.5,0,0,0],
               [0.1,0,0.5,0.3,0,0.1],
               [0,0,0,0.7,0.1,0.2],
               [1./3,0,0,1./3,1./3,0],
               [0,0,0,0,0,1]])


P_2 = np.array([[1,0,0,0,0,0],
                [0,3./4,1./4,0,0,0],
                [0,1./8,7./8,0,0,0],
                [1./4,1./4,0,1./8,3./8,0],
                [1./3,0,1./6,1./4,1./4,0],
                [0,0,0,0,0,1]])


T_1 = make_graph(P_1)

plt.figure()
plt.title("Directed Graph of Transition Probabilties")
pos = nx.circular_layout(T_1)
nx.draw(T_1, pos, node_size=1500, node_color="white", with_labels=True)
nx.draw_networkx_edges(T_1, pos, arrows=True)


T_2 = make_graph(P_2)

plt.figure()
plt.title("Directed Graph of Transition Probabilties")
pos = nx.circular_layout(T_2)
nx.draw(T_2, pos, node_size=1500, node_color="white", with_labels=True)
nx.draw_networkx_edges(T_2, pos, arrows=True)

plt.show()

