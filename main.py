"""Finding an optimal ski route in Tyrol. 
    Based on http://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/"""

import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def get_imbalanced_nodes(graph):
    """Finds all imbalanced nodes in a given directed graph. Returns a list of imbalanced nodes."""
    imbalanced_nodes_positive = []
    imbalanced_nodes_negative = []
    imbalanced_nodes = []
    count_positive = 1
    count_negative = 1
    for n in graph.nodes():
        if (graph.in_degree(n) - graph.out_degree(n)) > 0:
            print(
                f"Imbalanced positive node #{count_positive}: {n}, off by {graph.in_degree(n) - graph.out_degree(n)}"
            )
            count_positive += 1
            imbalanced_nodes_positive.append(n)
        elif (graph.in_degree(n) - graph.out_degree(n)) < 0:
            imbalanced_nodes_negative.append(n)
            print(
                f"Imbalanced negative node #{count_negative}: {n}, off by {graph.in_degree(n) - graph.out_degree(n)}"
            )
            count_negative += 1

    combinations = itertools.product(
        imbalanced_nodes_positive, imbalanced_nodes_negative
    )

    for combination in combinations:
        imbalanced_nodes.append(combination)

    return imbalanced_nodes


def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    """Compute shortest distance between each pair of nodes in a graph.  Return a dictionary keyed on node pairs (tuples)."""
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(
            graph, pair[0], pair[1], weight=edge_weight_name
        )
    return distances


def create_complete_graph(pair_weights, flip_weights=True):
    """
    Create a completely connected graph using a list of vertex pairs and the shortest path distances between them
    Parameters:
        pair_weights: list[tuple] from the output of get_shortest_paths_distances
        flip_weights: Boolean. Should we negate the edge attribute in pair_weights?
    """
    cg = nx.Graph()
    for k, v in pair_weights.items():
        wt_i = -v if flip_weights else v
        cg.add_edge(k[0], k[1], **{"distance": v, "weight": wt_i})
    return cg


def add_augmenting_path_to_graph(graph, min_weight_pairs):
    """
    Add the min weight matching edges to the original graph
    Parameters:
        graph: NetworkX graph (original graph from trailmap)
        min_weight_pairs: list[tuples] of node pairs from min weight matching
    Returns:
        augmented NetworkX graph
    """

    # We need to make the augmented graph a MultiGraph so we can add parallel edges
    graph_aug = nx.MultiDiGraph(graph.copy())
    for pair in min_weight_pairs:
        graph_aug.add_edge(
            pair[0],
            pair[1],
            **{
                "distance": nx.dijkstra_path_length(graph, pair[0], pair[1]),
                "trail": "augmented",
            },
        )
    return graph_aug


def create_eulerian_circuit(graph_augmented, graph_original, starting_node=None):
    """Create the eulerian path using only edges from the original graph."""
    euler_circuit = []
    naive_circuit = list(nx.eulerian_circuit(graph_augmented, source=starting_node))

    for edge in naive_circuit:
        edge_data = graph_augmented.get_edge_data(edge[0], edge[1])

        if edge_data[0]["trail"] != "augmented":
            # If `edge` exists in original graph, grab the edge attributes and add to eulerian circuit.
            edge_att = graph_original[edge[0]][edge[1]]
            euler_circuit.append((edge[0], edge[1], edge_att))
        else:
            aug_path = nx.shortest_path(
                graph_original, edge[0], edge[1], weight="distance"
            )
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

            print("Filling in edges for augmented edge: {}".format(edge))
            print("Augmenting path: {}".format(" => ".join(aug_path)))
            print("Augmenting path pairs: {}\n".format(aug_path_pairs))

            # If `edge` does not exist in original graph, find the shortest path between its nodes and
            #  add the edge attributes for each link in the shortest path.
            for edge_aug in aug_path_pairs:
                edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
                euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))

    return euler_circuit

#Need to make this dynamic - currently only works for St. Johann / Oberndorf
edgelist = pd.read_csv("edgelist_stjohann_tirol.csv")
nodelist = pd.read_csv("nodelist_stjohann_tirol.csv")
start_node = "ob_bauernalm_tal"

print(edgelist)
print(nodelist)
print("\n")

# Creating a directed graph as skilifts and slopes can only be done in one direction
g = nx.DiGraph()

# Adding all edges to the graph
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow["node1"], elrow["node2"], **elrow[2:].to_dict())

# Adding all node to the graph
for i, nlrow in nodelist.iterrows():
    nx.set_node_attributes(g, {nlrow["id"]: nlrow[1:].to_dict()})

print(f"Number of edges: {g.number_of_edges()}")
print(f"Number of nodes: {g.number_of_nodes()}")
print(f"Graph is strongly connected: {nx.is_strongly_connected(g)}")
print("\n")

# Negating the Y Axis to draw the graph "right way up"
node_positions = {node[0]: (node[1]["X"], -node[1]["Y"]) for node in g.nodes(data=True)}

# Defining the edge colors
edge_colors = [e[2]["color"] for e in list(g.edges(data=True))]

plt.figure(figsize=(8, 6))
nx.draw(g, pos=node_positions, edge_color=edge_colors, node_size=10, node_color="black")
plt.suptitle("Graph Representation of St. Johann / Oberndorf ski area", size=12)
plt.show()

# Obtaining imbalanced nodes
imbalanced_nodes = get_imbalanced_nodes(g)
print(f"These nodes are imbalanced:\n{imbalanced_nodes}\n")
imbalanced_nodes_shortest_paths = get_shortest_paths_distances(
    g, imbalanced_nodes, "distance"
)

print(
    f"These are the shortest paths of the imbalanced nodes:\n{imbalanced_nodes_shortest_paths}\n"
)

g_imbalanced_complete = create_complete_graph(
    imbalanced_nodes_shortest_paths, flip_weights=True
)

""" plt.figure(figsize=(8, 6))
pos_random = nx.random_layout(g_imbalanced_complete)
nx.draw_networkx_nodes(g_imbalanced_complete, node_positions, node_size=20, node_color="red")
nx.draw_networkx_edges(g_imbalanced_complete, node_positions, alpha=0.1)
plt.axis('off')
plt.title('Complete Graph of Odd-degree Nodes')
plt.show() """

# For some reason unbeknownst to me (and not answered by the author of the underlying tutorial),
# max_weight_matching mixes up starting & end nodes, which is an issue in a directed graph
# The below implementation should fix it; however, is likely to not be the most performant solution

imbalanced_matching = nx.algorithms.max_weight_matching(g_imbalanced_complete, True)

print("Number of edges in matching: {}".format(len(imbalanced_matching)))
print(f"Before adjusting: {imbalanced_matching}")

for match in imbalanced_matching.copy():
    for node in imbalanced_nodes:
        if node[0] == match[1] and node[1] == match[0]:
            imbalanced_matching.remove(match)
            imbalanced_matching.add(node)

print(f"After adjusting:  {imbalanced_matching}")

g_aug = add_augmenting_path_to_graph(g, imbalanced_matching)

print(f"Number of edges in original graph: {len(g.edges())}")
print(f"Number of edges in augmented graph: {len(g_aug.edges())}")
print(f"Number of imbalanced nodes in augmented graph: {get_imbalanced_nodes(g_aug)}")

# If there is one node with a higher difference than |1|, add it again; this is NOT an elegant solution
# Maybe one could solve this by adding a loop here. Also unclear if this is now truly the optimal solution?
if len(get_imbalanced_nodes(g_aug)) != 0:
    g_aug = add_augmenting_path_to_graph(g_aug, get_imbalanced_nodes(g_aug))
    print(f"Number of edges in augmented graph: {len(g_aug.edges())}")
    print(
        f"Number of imbalanced nodes in augmented graph: {get_imbalanced_nodes(g_aug)}"
    )

euler_circuit = create_eulerian_circuit(g_aug, g, start_node)

for step, edge in enumerate(euler_circuit, start=1):
    print(
        f"Step {step}: Start at {edge[0]}, go to {edge[1]} using {edge[2]['type_en']} {edge[2]['trail']}"
    )


total_time = sum(edge[2]["distance"] for edge in euler_circuit)

#Convert the value of total_time to a readable format which is hh:mm:ss. total_time is in seconds.
total_time = str(datetime.timedelta(seconds=total_time))
print(total_time)


# Creating a visual for the augmented, solved graph
def create_cpp_edgelist(euler_circuit):
    """
    Create the edgelist without parallel edge for the visualization
    Combine duplicate edges and keep track of their sequence and # of walks
    Parameters:
        euler_circuit: list[tuple] from create_eulerian_circuit
    """
    cpp_edgelist = {}

    for i, e in enumerate(euler_circuit):
        edge = frozenset([e[0], e[1]])

        if edge in cpp_edgelist:
            cpp_edgelist[edge][2]["sequence"] += ", " + str(i)
            cpp_edgelist[edge][2]["visits"] += 1

        else:
            cpp_edgelist[edge] = e
            cpp_edgelist[edge][2]["sequence"] = str(i)
            cpp_edgelist[edge][2]["visits"] = 1

    return list(cpp_edgelist.values())


cpp_edgelist = create_cpp_edgelist(euler_circuit)

g_cpp = nx.DiGraph(cpp_edgelist)

""" plt.figure(figsize=(8, 6))

visit_colors = {1:'lightgray', 2:'blue', 3:'darkorange'}
edge_colors = [visit_colors[e[2]['visits']] for e in g_cpp.edges(data=True)]
node_colors = ['red'  if node in imbalanced_nodes else 'lightgray' for node in g_cpp.nodes()]

nx.draw_networkx(g_cpp, pos=node_positions, node_size=20, node_color=node_colors, edge_color=edge_colors, with_labels=False)
plt.axis('off')
plt.suptitle('Lifts & Slopes to travel') """

plt.figure(figsize=(12, 8))

edge_colors = [e[2]["color"] for e in g_cpp.edges(data=True)]
nx.draw_networkx(
    g_cpp,
    pos=node_positions,
    node_size=10,
    node_color="black",
    edge_color=edge_colors,
    with_labels=False,
    alpha=0.5,
)

bbox = {
    "ec": [1, 1, 1, 0],
    "fc": [1, 1, 1, 0],
}  # hack to label edges over line (rather than breaking up line)
edge_labels = nx.get_edge_attributes(g_cpp, "sequence")
nx.draw_networkx_edge_labels(
    g_cpp, pos=node_positions, edge_labels=edge_labels, bbox=bbox, font_size=6
)

plt.axis("off")
plt.suptitle(
    "Overview of Trails & Lifts in Oberndorf/St.Johann Ski Area - in order of optimal travel"
)
plt.show()
