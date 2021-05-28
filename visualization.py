import matplotlib.pyplot as plt
import networkx as nx

color_list = ['#660000', '#003300', '#003333', '#003399', '#FF6633', '#FF99FF', '#99FFCC', '#00CCFF',
             '#FF9999', '#CCCCFF', '#00CCCC', '#66FF66', '#D3D3D3', '#6633CC', '#00FFFF', '#660000', '#000000',
             '#CCFFFF', '#005500', '#000077', '#99FFFF', '#888888', '#FF0000', '#CC99FF', '#FFFFCC', '#66CC66',
             '#33CC00', '#6600FF', '#CC6600', '#663300', '#00FF00', '#99CCFF', '#3399FF', '#0000FF']

color_grad = ['#D3D3D3', '#999999', '#99CCFF', '#66CCFF', '#3399FF', '#3366FF', '#0000FF', '#0000CC', '#000066']

id = 'facebook/' + '698'
circle_file = id + '.circles'
edge_file = id + '.edges'
node_file = id + ".feat"
result_file = 'result698_2.txt'


def read_feature():
	features = []
	result = []
	feature_sigs = []
	handle = open('result698theta_first.txt')
	for line in handle:
		line = line.split(', ')
		line = [round(float(i), 3) for i in line]
		features.append(line)
	for feature in features:
		feature_sigs.append([feature.index(i) for i in filter(lambda x: x > 0.6, feature[1:])])
	return feature_sigs


def read_node_feature():
	node_features = {}
	node_index = {}
	nf = open(node_file)
	l = 0
	for line in nf:
		line = line.split(" ")
		node_id = int(line[0])
		if node_id not in node_index:
			node_index[node_id] = l  # node_index[1]=0, ...
		else:
			continue
		a = []
		for i in range(1, len(line)):
			a.append(int(line[i]))  # read the whole line except for the node_id
		node_features[l] = a  # feature for every user
		l += 1
	nf.close()
	return node_features, node_index


def draw_origin():
	G = nx.Graph()
	G_new = nx.Graph()

	node_color = []
	handle = open(node_file)
	for line in handle:
		string = line.split()
		G.add_node(string[0])
		G_new.add_node(string[0])
		node_color.append('#D3D3D3')

	node_name = list(G.nodes())

	handle = open(edge_file)

	for line in handle:
		string = line.split()
		G.add_edge(string[0], string[1])

	cnt = 0
	handle = open(circle_file)
	for line in handle:
		line = line.strip()
		line = line.split("\t")
		node_color[node_name.index(line[1])] = color_list[cnt]
		if len(line) == 2:
			cnt += 1
			continue
		for i in range(2, len(line)):
			G_new.add_edge(line[i - 1], line[i])
			node_color[node_name.index(line[i])] = color_list[cnt]
		G_new.add_edge(line[1], line[len(line) - 1])
		cnt += 1

	plt.subplot(1, 2, 1)
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=3)
	nx.draw_networkx_edges(G, pos, style='solid')
	plt.subplot(1, 2, 2)
	nx.draw_networkx_nodes(G_new, pos, node_color=node_color, node_size=5)
	nx.draw_networkx_edges(G_new, pos, style='dotted')
	nx.draw(G, node_color=node_color, node_size=3)
	plt.show()

	# draw separately
	feature_sigs = read_feature()
	node_features, node_index = read_node_feature()
	print(feature_sigs)

	handle = open(result_file)
	cnt = 0
	circles = [[] for i in range(len(feature_sigs[cnt]))]
	for line in handle:
		G_result = nx.Graph()
		node = line.split(", ")

		for i in range(len(node) - 1):
			G_result.add_node(node_name[int(node[i])])
			G_result.add_node(node_name[int(node[i + 1])])

		print(feature_sigs[cnt])
		for feature in feature_sigs[cnt]:
			for node in G_result:
				if node_features[node_index[int(node)]][feature] == 1:
					circles[feature_sigs[cnt].index(feature)].append(node)

		print(circles)
		for circle in circles:
			if len(circle) == 0:
				continue
			for i in range(len(circle) - 1):
				G_result.add_edge(circle[i], circle[i + 1])
			G_result.add_edge(circle[0], circle[len(circle) - 1])

		pos = nx.spring_layout(G_new)
		plt.subplot(1, 2, 1)
		nx.draw_networkx_nodes(G_new, pos, node_color=node_color, node_size=5)
		nx.draw_networkx_edges(G_new, pos, style='dotted')

		plt.subplot(1, 2, 2)
		nx.draw_networkx_nodes(G_result, pos, node_color='#D3D3D3', node_size=5)
		nx.draw_networkx_edges(G_result, pos, style='dotted')
		# nx.draw(G_result, pos, node_color='#D3D3D3', node_size=5)
		plt.show()
		cnt += 1

	# draw in one figure
	G_result2 = nx.Graph()
	node_dict = {}
	node_color_result = []

	handle = open(result_file)
	for line in handle:
		node = line.split(", ")
		for i in range(len(node)):
			node_dict[node_name[int(node[i])]] = node_dict.get(node_name[int(node[i])], 0) + 1

	for key in node_dict:
		G_result2.add_node(key)
		node_color_result.append('#D3D3D3')

	node_name_result = list(G_result2.nodes())
	for node in G_result2:
		node_color_result[node_name_result.index(node)] = color_grad[node_dict[node] - 1]

	pos = nx.spring_layout(G_new)
	plt.subplot(1, 2, 1)
	nx.draw_networkx_nodes(G_new, pos, node_color=node_color, node_size=5)
	nx.draw_networkx_edges(G_new, pos, style='dotted')

	plt.subplot(1, 2, 2)
	nx.draw_networkx_nodes(G_result2, pos, node_color=node_color_result, node_size=5)
	plt.show()


draw_origin()
