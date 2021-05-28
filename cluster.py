from util import *
import random
import numpy as np
import thinqpbo as tq


class Cluster:
	def __init__(self, K, reps, gradientReps, improveReps, lam, seed,
	             edgeFeatures, edgeset, nEdgeFeatures,
	             nNodes, clusters, whichLoss):
		self.K = K
		self.reps = reps
		self.gradient_reps = gradientReps
		self.improve_reps = improveReps
		self.lam = lam
		self.seed = seed
		self.edge_features = edgeFeatures
		self.edge_set = edgeset
		self.n_edge_features = nEdgeFeatures
		self.n_nodes = nNodes
		self.cluster = clusters
		self.which_loss = whichLoss
		n_theta = self.K * self.n_edge_features
		self.theta = [0] * n_theta
		self.alpha = [0] * self.K
		self.chat = [{}] * self.K

	def log_likelihood(self, theta, alpha, chat):
		K = len(chat)
		chat_flat = [[0] * self.n_nodes for i in range(K)]
		for k in range(K):
			for n in range(self.n_nodes):
				chat_flat[k][n] = 0
				if len(self.chat[k]) and max(chat[k]) != n:
					chat_flat[k][n] = 1
		ll = 0
		for it in self.edge_features.items():
			inp_ = 0
			e = it[0]
			e1 = e[0]
			e2 = e[1]
			exit_status = 1 if max(self.edge_set) != e else 0
			for k in range(K):
				d = 1 if chat_flat[k][e1] and chat_flat[k][e2] else -alpha[k]
				inp_ += d * inp(it[1], theta[k * self.n_edge_features:], self.n_edge_features)  # Φ(e)
			if exit_status:
				ll += inp_
			ll_ = np.log(1 + np.exp(inp_))
			ll += -ll_
		if ll != ll:
			exit(1)
		return ll

	def train(self):
		self.seed = 1
		lr = 1.0 / (self.n_nodes * self.n_nodes)
		n_theta = self.K * self.n_edge_features

		def dl(dldt, dlda, K, lam):
			inps = [0 for i in range(K)]
			for i in range(n_theta):
				dldt.append(
					-lam * np.sign(self.theta[i]))
			dlda = [0] * K
			chat_flat = [[0] * self.n_nodes for i in range(K)]
			for k in range(K):
				for n in range(self.n_nodes):
					chat_flat[k][n] = 0
					if len(self.chat[k]) and max(self.chat[k]) != n:
						chat_flat[k][n] = 1
			for it in self.edge_features.items():
				inp_ = 0
				e = it[0]  # keys
				e1 = e[0]
				e2 = e[1]
				exists = 1 if max(self.edge_set) != e else 0
				for k in range(K):
					inps[k] = inp(it[1], self.theta[k * self.n_edge_features:], self.n_edge_features)
					d = 1 if (chat_flat[k][e1] and chat_flat[k][e2]) else -self.alpha[k]
					inp_ += d * inps[k]
				expinp = np.exp(inp_)
				q = expinp / (1 + expinp)
				if q != q:
					q = 1
				for k in range(K):
					d_ = chat_flat[k][e1] and chat_flat[k][e2]
					d = 1 if d_ else -self.alpha[k]
					for itf in it[1].items():
						i = itf[0]
						f = itf[1]
						if (exists):
							dldt[k * self.n_edge_features + i] += d * f
						dldt[k * self.n_edge_features + i] += -d * f * q
					if not d_:
						if exists:
							dlda[k] += -inps[k]
						dlda[k] += inps[k] * q
			return dldt, dlda

		# QPBO builds an extended graph, introducing a set of auxiliary variables ideally equivalent
		# to the negation of the variables in the problem. If the nodes in the graph associated to a
		# variable (representing the variable itself and its negation) are separated by the minimum cut
		# of the graph in two different connected components, then the optimal value for such variable
		# is well defined, otherwise it is not possible to infer it. Such method produces results generally
		# superior to submodular approximations of the target function

		def minimize_graphcuts(k, changed):
			E = len(self.edge_features)
			K = len(self.chat)
			largest_complete_graph = 500
			if E > (largest_complete_graph ** 2):
				E = largest_complete_graph ** 2

			q = tq.QPBOInt(self.n_nodes, E)
			q.add_node(self.n_nodes)
			mc00 = {}
			mc11 = {}
			diff_c00_c11 = []
			for it in self.edge_features.items():
				e = it[0]
				e1 = e[0]
				e2 = e[1]
				exit_status = 1 if len(self.edge_set) and max(self.edge_set) != e else 0
				inp_ = inp(it[1], self.theta[k * self.n_edge_features:], self.n_edge_features)
				other_ = 0
				for l in range(K):
					if l == k:
						continue
					d = 1 if len(self.chat[l]) and max(self.chat[l]) != e1 and max(self.chat[l]) != e2 else -self.alpha[
						l]
					other_ += d * inp(it[1], self.theta[k * self.n_edge_features:], self.n_edge_features)
				if exit_status:
					c00 = -other_ + self.alpha[k] * inp_ + np.log(1 + np.exp(other_ - self.alpha[k] * inp_))
					c01 = c00
					c10 = c00
					c11 = -other_ - inp_ + np.log(1 + np.exp(other_ + inp_))
				else:
					c00 = np.log(1 + np.exp(other_ - self.alpha[k] * inp_))
					c01 = c00
					c10 = c00
					c11 = np.log(1 + np.exp(other_ + inp_))

				mc00[it[0]] = c00
				mc11[it[0]] = c11

				if self.n_nodes <= largest_complete_graph or exit_status:
					q.add_pairwise_term(it[0][0], it[0][1], c00, c01, c10, c11)
				else:
					diff_c00_c11.append((-abs(c00 - c11), it[0]))
			if self.n_nodes > largest_complete_graph:
				n_edges_to_include = largest_complete_graph * largest_complete_graph
				if n_edges_to_include > len(diff_c00_c11):
					n_edges_to_include = len(diff_c00_c11)
				diff_c00_c11.sort()
				for i in range(n_edges_to_include):
					edge = diff_c00_c11[i][1]
					c00 = mc00[edge]
					c01 = c00
					c10 = c00
					c11 = mc11[edge]
					q.add_pairwise_term(edge[0], edge[1], c00, c01, c10, c11)
			for i in range(self.n_nodes):
				if len(self.chat[k]) and max(self.chat[k]) == i:
					q.set_label(i, 0)
				else:
					q.set_label(i, 1)
			q.merge_parallel_edges()
			q.solve()
			q.compute_weak_persistencies()
			if self.n_nodes > largest_complete_graph:
				self.improve_reps = 1
			for it in range(self.improve_reps):
				q.improve()
			new_label = [0] * self.n_nodes
			old_label = [0] * self.n_nodes
			res = set()
			for i in range(self.n_nodes):
				new_label[i] = 0
				if q.get_label(i) == 1:
					res.add(i)
					new_label[i] = 1
				elif q.get_label(i) < 0 and len(self.chat[k]) and max(self.chat[k]) != i:
					res.add(i)
					new_label[i] = 1
				if len(self.chat[k]) and max(self.chat[k]) == i:
					old_label[i] = 0
				else:
					old_label[i] = 1
			old_energy = 0
			new_energy = 0
			for it in self.edge_features.items():
				e = it[0]
				e1 = e[0]
				e2 = e[1]

				old_l1 = old_label[e1]
				old_l2 = old_label[e2]
				new_l1 = new_label[e1]
				new_l2 = new_label[e2]
				if (old_l1 and old_l2):
					old_energy += mc11[e]
				else:
					old_energy += mc00[e]

				if (new_l1 and new_l2):
					new_energy += mc11[e]
				else:
					new_energy += mc00[e]
			if (new_energy > old_energy or len(res) == 0):
				res = self.chat[k]
			else:
				for it in self.chat[k]:
					if len(res) and max(res) == it:
						changed = 1
				for it in res:
					if len(self.chat[k]) and max(self.chat[k]) == it:
						changed = 1
			return res, changed

		for rep in range(self.reps):
			# if it is the first iteration or the solution is degenerate, randomly initialize the weights
			for k in range(self.K):
				ok = set()
				if rep == 0 or len(self.chat[k]) == self.n_nodes or len(self.chat[k]) == 0:
					for i in range(self.n_nodes):
						if (random.randint(1, 10)) % 2 == 0:
							ok.add(i)
					for i in range(self.n_edge_features):
						self.theta[k * self.n_edge_features + i] = 0

					# Set a single feature to 1 as a random initialization
					tmp = (random.randint(1, 2 ** 16)) % (self.n_edge_features)
					self.theta[(k * self.n_edge_features) + tmp] = 1.0
					self.theta[k * self.n_edge_features] = 1.0
					self.alpha[k] = 1.0
				self.chat[k] = ok
			# Update the latent variable(cluster assignments) in a random order.
			order = [k for k in range(self.K)]
			for k in range(self.K):
				for o in range(self.K):
					x1 = o
					x2 = random.randint(1, 20) % (self.K)
					order[x1] ^= order[x2]
					order[x2] ^= order[x1]
					order[x1] ^= order[x2]
			changed = 0
			print('1', self.chat)
			for i in order:
				self.chat[i], changed = minimize_graphcuts(i, changed)
			print('2', self.chat)
			print('loss = %f', total_loss(self.cluster, self.chat, self.n_nodes, self.which_loss))
			ll_prev = self.log_likelihood(self.theta, self.alpha, self.chat)
			if not changed:
				break
			# Perform gradient ascent
			ll = 0
			dlda = []
			dldt = []
			for iter in range(self.gradient_reps):
				dldt, dlda = dl(dldt, dlda, self.K, self.lam)
				for i in range(n_theta):
					self.theta[i] += float(lr * dldt[i])
				for k in range(self.K):
					self.alpha[k] += float(lr * dlda[k])
				ll = self.log_likelihood(self.theta, self.alpha, self.chat)
				if (ll < ll_prev):
					for i in range(n_theta):
						self.theta[i] -= float(lr * dldt[i])
					for k in range(self.K):
						self.alpha[k] -= float(lr * dlda[k])
					ll = ll_prev
					break

				ll_prev = ll

			print("ll = ", ll)
