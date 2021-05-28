from util import *


class GraphData:
    def __init__(self, node_file, self_feat_file, clus_file, edge_file, which):
        self.nf = node_file
        self.ff = self_feat_file
        self.cf = clus_file
        self.ef = edge_file
        self.ftype = which

    def file_process(self):
        sim_features = {}
        clusters = []
        node_features = {}
        node_index = {}

        nf = open(self.nf)
        l = 0
        for line in nf:
            line = line.split(" ")
            node_id = int(line[0])
            if node_id not in node_index:
                node_index[node_id] = l
            else:
                continue
            a = []
            for i in range(1, len(line)):
                a.append(int(line[i]))
            node_features[l] = a
            l += 1
        n_nodes = len(node_index.keys())  # number of alters
        n_node_features = len(a)  # number of features
        nf.close()

        ff = open(self.ff)
        for line in ff:
            line = line.strip()
            line = line.split(" ")
        sf = [int(i) for i in line]
        for i in range(n_nodes):
            res = diff(sf, node_features[i], n_node_features)
            sim_features[i] = res  # 0 similar 1 different
        ff.close()

        cf = open(self.cf)
        for line in cf:
            circle = set()
            line = line.strip()
            line = line.split("\t")
            for i in range(1, len(line)):
                node_id = int(line[i])
                circle.add(node_index[node_id])
            clusters.append(circle)
        cf.close()

        n_edge_features = 1 + n_node_features
        if self.ftype == 'BOTH':
            n_edge_features += n_node_features
        edge_features = {}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                d = [1]
                if self.ftype == 'EGO':
                    d = d + diff(sim_features[i], sim_features[j], n_node_features)
                elif (self.ftype == 'FRIEND'):
                    d = d + diff(node_features[i], node_features[j], n_node_features)
                else:
                    d = d + (diff(sim_features[i], sim_features[j], n_node_features))
                    d = d + (diff(node_features[i], node_features[j], n_node_features))
                edge_features[(i, j)] = sparse(d, n_edge_features)

        # read edge
        ef = open(self.ef)
        edge_set = set()
        for line in ef:
            line = line.split(' ')
            id1 = int(line[0])
            id2 = int(line[1])
            index1 = node_index[id1]
            index2 = node_index[id2]
            edge_set.add((index1, index2))
        ef.close()

        return edge_features, edge_set, n_edge_features, n_nodes, clusters, node_index

