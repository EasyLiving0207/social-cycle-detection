from cluster import Cluster
from graph import GraphData
from util import *


def experiment(gd, K, lam, reps, gradient_reps, improve_reps, file_name):
	best_ll = 0
	best_clusters = []
	best_theta = 0
	best_alpha = 0
	seed = 42
	edge_features, edge_set, n_edge_features, n_nodes, clusters, node_index = gd.file_process()

	C = Cluster(K, reps, gradient_reps, improve_reps, lam, seed, edge_features,
	            edge_set, n_edge_features, n_nodes, clusters, whichLoss='SYMMETRICDIFF')
	n_seeds = 1  # Number of random restarts
	for seed in range(n_seeds):
		seed += 1
		C.train()
		ll = C.log_likelihood(C.theta, C.alpha, C.chat)
		if ll > best_ll or best_ll == 0:
			best_ll = ll
			best_clusters = C.chat
			best_theta = C.theta
			best_alpha = C.alpha

	file = open(file_name, 'w')
	print('ll = ', best_ll, file=file)
	print('loss_zeroone = ', total_loss(clusters, best_clusters, n_nodes, 'ZEROONE'), file=file)
	print("loss_symmetric = ", total_loss(clusters, best_clusters, n_nodes, 'SYMMETRICDIFF'), file=file)
	print("fscore = ", 1 - total_loss(clusters, best_clusters, n_nodes, 'FSCORE'), file=file)
	print('Clusters:\n', best_clusters, file=file)
	print('Theta:\n', best_theta, file=file)
	print('Alpha:\n', best_alpha, file=file)
	file.close()


ID = 'facebook/' + '698'
node_file = ID + ".feat"
self_feat_file = ID + '.egofeat'
clus_file = ID + '.circles'
edge_file = ID + '.edges'

gd = GraphData(node_file, self_feat_file, clus_file, edge_file, 'FRIEND')
experiment(gd, 12, 1, 25, 50, 5, 'result' + ID + '.txt')
