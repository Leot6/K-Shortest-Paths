"""
Yen's k shortest paths algorithm. The network is built on networkx.
"""

import copy
import time
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from itertools import islice

with open('NYC_NET.pickle', 'rb') as f:
    NYC_NET = pickle.load(f)
G = copy.deepcopy(NYC_NET)


# # parameters for Manhattan map
# map width and height (km)
MAP_WIDTH = 10.71
MAP_HEIGHT = 20.85
# coordinates
# (Olng, Olat) lower left corner
Olng = -74.0300
Olat = 40.6950
# (Olng, Olat) upper right corner
Dlng = -73.9030
Dlat = 40.8825


# return the travel time of edge (u, v)
def get_edge_dur(u, v):
    return NYC_NET.get_edge_data(u, v, default={'dur': None})['dur']


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
                root_path_duration += get_edge_dur(u, v)

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
                        edges_removed.append((u, v,get_edge_dur(u, v)))
                        # print('u, v, edge_duration (remove)', u, v, edge_duration)

            # remove rootPathNode (except spurNode) from Graph
            for n in range(len(root_path) - 1):
                u = root_path[n]
                # print('node', u)
                # out-edges
                nodes = copy.deepcopy(G[u])
                for v in nodes:
                    G.remove_edge(u, v)
                    edges_removed.append((u, v, get_edge_dur(u, v)))
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
    KSP = []
    for (dur, path) in A:
        KSP.append(path)
    return KSP


# k-shortest paths algorithm in networkx
def k_shortest_paths_nx(G, source, target, k, weight='dur'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def plot_path(onid, dnid, paths):
    fig = plt.figure(figsize=(MAP_WIDTH, MAP_HEIGHT))
    plt.xlim((Olng, Dlng))
    plt.ylim((Olat, Dlat))
    img = mpimg.imread('map.png')
    plt.imshow(img, extent=[Olng, Dlng, Olat, Dlat], aspect=(Dlng - Olng) / (Dlat - Olat) * MAP_HEIGHT / MAP_WIDTH)
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00)
    # [olng, olat] = G.nodes[onid]['pos']
    # [dlng, dlat] = G.nodes[dnid]['pos']
    # plt.scatter(olng, olat)
    # plt.scatter(dlng, dlat)
    for index, path in zip(range(len(paths)), paths):
        x = []
        y = []
        for node in path:
            [lng, lat] = NYC_NET.nodes[node]['pos']
            x.append(lng)
            y.append(lat)
        if index == 0 or index == 1:
            plt.plot(x, y, marker='.')
        else:
            plt.plot(x, y, '--')

    plt.savefig('example.jpg', dpi=300)
    plt.show()


if __name__ == "__main__":
    onid = 800
    dnid = 2300
    aa = time.time()
    # KSP = k_shortest_paths(G, onid, dnid, 20, 'dur')
    KSP1 = k_shortest_paths_nx(NYC_NET, onid, dnid, 20, 'dur')
    print('find k shortest paths running time:', (time.time() - aa))
    plot_path(onid, dnid, KSP1)

