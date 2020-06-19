"""
Yen's k shortest paths algorithm. The network is built on networkx.
"""

import math
import copy
import networkx as nx
from itertools import islice

with open('NYC_NET.pickle', 'rb') as f:
    NYC_NET = pickle.load(f)
G = copy.deepcopy(NOD_NET)


# returns the k-shortest paths from source to target in a weighted graph G
def k_shortest_paths(G, source, target, k=1, weight='dur'):
    # Determine the shortest path from the source to the target
    duration, path = nx.bidirectional_dijkstra(G, source, target, weight=weight)
    A = [tuple([duration, path])]  # k_shortest_paths
    B = []
    for i in range(1, k):
        i_path = A[-1][1]  # k-1 shortest path
        #  The spur node ranges from the first node to the next to last node in the previous k-shortest path
        for j in range(len(i_path) - 1):
            # Spur node is retrieved from the previous k-shortest path, k − 1.
            spur_node = i_path[j]
            root_path = i_path[:j + 1]

            root_path_duration = 0
            for u_i in range(len(root_path) - 1):
                u = root_path[u_i]
                v = root_path[u_i + 1]
                root_path_duration += get_edge_real_dur(u, v)

            # print('root_path', root_path)
            # print('root_path_duration', root_path_duration)

            edges_removed = []
            for path_k in A:
                curr_path = path_k[1]
                # Remove the links that are part of the previous shortest paths which share the same root path
                if len(curr_path) > j and root_path == curr_path[:j + 1]:
                    u = curr_path[j]
                    v = curr_path[j + 1]
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v,get_edge_real_dur(u, v)))
                        # print('u, v, edge_duration (remove)', u, v, edge_duration)

            # remove rootPathNode (except spurNode) from Graph
            for n in range(len(root_path) - 1):
                u = root_path[n]
                # print('node', u)
                # out-edges
                nodes = copy.deepcopy(G[u])
                for v in nodes:
                    G.remove_edge(u, v)
                    edges_removed.append((u, v, get_edge_real_dur(u, v)))
                    # print('u, v, edge_duration (remove)', u, v, edge_duration)
                # if G.is_directed():
                #     # in-edges
                #     for u, v, edge_duration in G.in_edges_iter(node, data=True):
                #         print('u, v, edge_duration (in)', u, v, edge_duration)
                #         G.remove_edge(u, v)
                #         edges_removed.append((u, v, edge_duration))

            try:
                # Calculate the spur path from the spur node to the target
                spur_path_duration, spur_path = nx.bidirectional_dijkstra(G, spur_node, target, weight='dur')
                # Entire path is made up of the root path and spur path
                total_path = root_path[:-1] + spur_path
                total_path_duration = root_path_duration + spur_path_duration
                potential_k = tuple([total_path_duration, total_path])
                # Add the potential k-shortest path to the heap
                if potential_k not in B:
                    B.append(potential_k)
                    # print('potential_k', potential_k)
            except nx.NetworkXNoPath:
                # print('NetworkXNoPath')
                pass

            # Add back the edges and nodes that were removed from the graph
            for u, v, edge_duration in edges_removed:
                G.add_edge(u, v, weight=edge_duration)
                # print('u, v, edge_duration (add)', u, v, edge_duration)

        if len(B):
            B.sort(key=lambda e: e[0])
            A.append(B[0])
            B.pop(0)
        else:
            break
    A.sort(key=lambda p: p[0])
    return A


def k_shortest_paths_nx(source, target, k, weight='dur'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))






