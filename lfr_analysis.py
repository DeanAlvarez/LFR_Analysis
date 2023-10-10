import networkx as nx
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import numpy as np

# Function for getting the size of the community in terms of nodes and edges.
# Inputs:
#   - dir: Path to the input file containing community information.
#   - split: Optional parameter specifying the delimiter for splitting values in the input file (default is space " ").
# Outputs:
#   - Returns the number of unique nodes and edges in the community.
def graph_size_from_file(dir, split=" "):
    nodes = set()
    edges = 0
    with open(dir, "r") as fp:
        for i in fp.readlines():
            i = i.strip()
            if "-1" not in i and i:
                a , b = i.split(split)
                nodes.add(a)
                nodes.add(b)
                edges += 1
            else:
                break

    return len(nodes), edges

# Helper function to read true community assignments from a file.
# Inputs:
#   - dir: Path to the file containing true community assignments.
# Outputs:
#   - Returns dictionaries mapping nodes to communities and communities to nodes.
def true_community_helper(dir):
    node_to_com = {}
    com_to_nodes = {}
    with open(dir, "r") as fp:
        for i in fp.readlines():
            line = i.strip()
            if i:
                a , b = line.split("\t")
                a = int(a)
                b = int(b)
                node_to_com[a] = b
                if b in com_to_nodes.keys():
                    com_to_nodes[b].add(a)
                else:
                    com_to_nodes[b] = set([a])
                
            else:
                break

    return node_to_com, com_to_nodes

# Function to get nodes in the true community of a given node.
# Inputs:
#   - node_to_com: Dictionary mapping nodes to communities.
#   - com_to_nodes: Dictionary mapping communities to nodes.
#   - q_node: Node for which you want to find the true community.
# Outputs:
#   - Returns the set of nodes in the true community of the input node.
def get_true_community(node_to_com, com_to_nodes, q_node):
    com = node_to_com[q_node]
    return com_to_nodes[com]

# Function to calculate error statistics for community detection.
# Inputs:
#   - dir: Path to the input file containing proposed community assignments.
#   - true_com: Set of nodes representing the true community.
#   - LFR: total network size, default is 10k for LFR10k
# Outputs:
#   - Returns symmetric difference, precision, recall, F1-score, and the size of the proposed community.
def LFR_error_stats(dir, true_com, LFR=10000):
    prop_com = set()
    with open(dir, "r") as fp:
        for i in fp.readlines():
            if "-1" not in i:
                a , b = i.strip().split(" ")
                a = int(a)
                b = int(b)
                prop_com.add(a)
                prop_com.add(b)
            else:
                break
    
    sym_diff = true_com.symmetric_difference(prop_com)
    full_net = set(range(1, LFR+1)) 

    true_positive = true_com.intersection(prop_com)
    true_negative = full_net.difference(true_com, prop_com)
    false_negative = true_com.difference(prop_com)
    false_positive = prop_com.difference(true_com)

    if len(prop_com) == 0:
        return sym_diff, 0, 0, 0, 0

    per = len(true_positive) / (len(true_positive) + len(false_positive))
    rec = len(true_positive) / (len(true_positive) + len(false_negative))

    f1 = (2 * per * rec) / (per + rec)

    return sym_diff, per, rec, f1, len(prop_com)

if __name__ == "__main__":
    # Configuration parameters
    lfr_param = "05"
    query_nodes = [9525, 9002, 8808, 6789]
    ks = range(10, 26)
    true_d = "data/LFR_10000/0.{}/community.txt".format(lfr_param)
    n_t_c, c_t_n = true_community_helper(true_d)
    n_data = {n: [[], [], [], []] for n in query_nodes}

    # Loop through nodes and k values to calculate statistics
    for n in query_nodes:
        for k in ks:
            com_d = "data/com_search/LFR0{}/{}/kcore_k{}.txt".format(lfr_param, n, k)
            true_com = get_true_community(n_t_c, c_t_n, n)
            sym, per, rec, f1, prop_com_size = LFR_error_stats(com_d, true_com)
            n_data[n][0].append(per)
            n_data[n][1].append(rec)
            n_data[n][2].append(f1)
            n_data[n][3].append((len(true_com), prop_com_size))
            print("k={} | precision: {} recall: {} f1: {}".format(k, per, rec, f1))

    # Plotting results
    fig, axs = plt.subplots(2, 2)
    plt.suptitle("k vs Precision for k-core on LFR0.05")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axs.flat):
        n = query_nodes[i]
        ax.plot(ks, n_data[n][0])
        ax.set_title("node: {}".format(n))

    plt.savefig("lfr0{}_output".format(lfr_param))
