import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
G = nx.Graph()

# Connect nodes 1 to 10 in a circle (circular backbone)
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)])

# Connect all even-numbered nodes (2, 4, 6, 8, 10)
even_nodes = [2, 4, 6, 8, 10]
G.add_edges_from([(i, j) for i in even_nodes for j in even_nodes if i != j])

# Connect all odd-numbered nodes (1, 3, 5, 7, 9)
odd_nodes = [1, 3, 5, 7, 9]
G.add_edges_from([(i, j) for i in odd_nodes for j in odd_nodes if i != j])

# Draw the graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
plt.show()
