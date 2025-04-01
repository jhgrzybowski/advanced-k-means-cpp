# main.py
import networkx as nx
from advanced_k_means import advanced_kmeans  # Import the function

# Create a sample network
G = nx.Graph()
edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('C', 'D'), ('D', 'E')]
G.add_edges_from(edges)

# Call the imported function
controllers = advanced_kmeans(G, K=3)
print("Optimal Controllers:", controllers)