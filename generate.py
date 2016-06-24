import numpy as np
from collections import OrderedDict
import sys, os

class ImageNode:
    def __init__(self, image_id, labels, features=tuple(), observed=True):
        self.image_id = image_id
        self.labels = labels
        self.features = tuple(features)
        self.observed = observed

    def get_id(self):
        return self.image_id

    def get_labels(self):
        return self.labels

    def get_features(self):
        return self.features

    def set_features(self, features):
        self.features = features

    def is_observed(self):
        return self.observed


class ImageGraph:
    def __init__(self):
        self.node_ids = []

        #node table; id -> node object
        self.nodes = {}

        #adjacency table; id -> [neighbor id's]
        self.adjacency = None
        self.id2ind = {}
        self.counter = 0

    def add_node(self, node):
        if (node.get_id() in self.node_ids):
            return #don't want to add nodes multiple times

        self.node_ids.append(node.get_id())
        self.nodes[node.get_id()] = node
        self.id2ind[node.get_id()] = self.counter
        self.counter += 1

    def load_edges(self, npy_filepath):
        self.adjacency = np.load(npy_filepath)

    def add_edge(self, id1, id2):
        if (self.adjacency == None):
            self.adjacency = np.zeros((len(self.node_ids), len(self.node_ids)))

        ind1 = self.id2ind[id1]
        ind2 = self.id2ind[id2]

        self.adjacency[ind1, ind2] = 1
        self.adjacency[ind2, ind1] = 1

    def get_neighbors(self, image_id):
        return self.adjacency[self.id2ind[image_id], :]

    def get_node(self, image_id):
        return self.nodes[image_id]

    def get_node_ids(self):
        return self.node_ids

    def num_nodes(self):
        return len(self.node_ids)

    def num_edges(self):
        return int(sum(sum(self.adjacency)))


class Data:
    def __init__(self):
        self.graph = ImageGraph()
        self.num_labels = None
        self.num_features = None

        self.feature_matrix = None
        self.edge_matrix = None
        self.label_matrix = None

    def add_nodes(self, image_filepath, feature_filepath):
        image_file = file(image_filepath, 'r')

        nodes = OrderedDict()

        print "reading nodes"
        i = 0 #small graph
        for line in image_file:
            split_line = line.split(' ') #split on spaces
            image_id = int(split_line[0])
            label_strings = split_line[-1]
            labels = map(int, label_strings.split(','))

            nodes[image_id] = ImageNode(image_id, labels)

            if self.num_labels == None:
                self.num_labels = len(labels)

            i += 1 #small graph
            # if i >= 10:
            #     break

        image_file.close()

        feature_file = file(feature_filepath, 'r')
        print "reading features"
        for line in feature_file:
            split_line = line.split(' ')
            image_id = int(split_line[0])
            num_features = int(split_line[1])
            features = map(float, split_line[2:2+num_features])

            if image_id in nodes:
                nodes[image_id].set_features(features)

            if self.num_features == None:
                self.num_features = num_features

        feature_file.close()

        for key in nodes.keys():
            self.graph.add_node(nodes[key])


    def add_edges(self, edge_filepath):
        print "Adding edges"
        global x
        edge_file = file(edge_filepath, 'r')

        for line in edge_file:
            split_line = line.split(' ')
            id1 = int(split_line[0])
            id2 = int(split_line[1])
            if id1 in self.graph.nodes and id2 in self.graph.nodes:
                self.graph.add_edge(id1, id2)

    def load_edges(self, npy_filepath):
        self.graph.load_edges(npy_filepath)

    def build_matrices(self):
        self.feature_matrix = np.zeros([self.graph.num_nodes(), self.num_features])
        self.label_matrix = np.zeros([self.graph.num_nodes(), 1+self.num_labels])
        #self.edge_matrix = np.zeros([self.graph.num_edges()/2, 2])
        self.edge_matrix = np.zeros([self.graph.num_edges(), 2])
        i = 0

        print "Building feature and label matrices"

        for node_id in self.graph.node_ids:
            image_node = self.graph.get_node(node_id)
            node_features = np.array(image_node.get_features())
            self.feature_matrix[i,:] = node_features/(np.linalg.norm(node_features)*6000)

            self.label_matrix[i,0] = node_id
            self.label_matrix[i,1:] = image_node.labels
            i += 1

        print "Building edge matrix"

        edge_ind = 0
        for i in range(self.graph.num_nodes()):
            for j in range(self.graph.num_nodes()):
                if (self.graph.adjacency[i,j] ):
                    self.edge_matrix[edge_ind,:] = [i, j]
                    edge_ind += 1

    def save_matrices(self, output_dir):
        np.savetxt(os.path.join(output_dir, 'features.csv'), self.feature_matrix, delimiter=' ')
        np.savetxt(os.path.join(output_dir, 'edges.csv'), self.edge_matrix, delimiter=' ')
        np.savetxt(os.path.join(output_dir, 'labels.csv'), self.label_matrix, delimiter=' ')

    def print_train_code(self, label_ind, model_file, output_file):
        with open(output_file, 'w') as out:
            print 'copying file'
            with open(model_file, 'r') as model:
                for line in model:
                    out.write(line)
            out.write('\n')
            print 'adding distinct statements'
            out.write('distinct Image Image[{0}];\n'.format(self.graph.num_nodes()))
            out.write('distinct Edge Edge[{0}];\n'.format(self.edge_matrix.shape[0]))
            out.write('\n')
            print 'adding image labels'
            for i in range(self.graph.num_nodes()):
                l = self.graph.get_node(self.graph.node_ids[i]).labels[label_ind]
                if l in [0,1]:
                    out.write('obs ImageLabel(Image[{0}]) = {1};\n'.format(i, float(l)))
            out.write('\n')
            print 'adding edges'
            i = 0
            out.write('obs ImagePair(Edge[{0}]) = 1;\n'.format(i))
            out.write('\n')
            print 'adding queries'
            out.write('query w;\n')
            out.write('query v1;\n')
            out.write('query v2;\n')

    def print_eval_code(self, label_ind, model_file, output_file):
        with open(output_file, 'w') as out:
            w = np.loadtxt('weights/w_{0}.csv'.format(label_ind))
            v1 = np.loadtxt('weights/v1_{0}.csv'.format(label_ind))
            v2 = np.loadtxt('weights/v2_{0}.csv'.format(label_ind))

            out.write('fixed RealMatrix w = [' + '; '.join([str(x) for x in w]) + '];\n')
            out.write('fixed RealMatrix v1 = [' + '; '.join([str(x) for x in v1]) + '];\n')
            out.write('fixed RealMatrix v2 = [' + '; '.join([str(x) for x in v2]) + '];\n')
            out.write('\n')

            with open(model_file, 'r') as model:
                for line in model:
                    out.write(line)
            out.write('\n')

            out.write('distinct Image Image[{0}];\n'.format(self.graph.num_nodes()))
            out.write('distinct Edge Edge[{0}];\n'.format(self.edge_matrix.shape[0]))
            out.write('\n')

            out.write('obs ImagePair(x) = 1 for Edge x;\n')
            out.write('obs jitter = 0.0;\n')
            out.write('\n')

            for i in range(self.graph.num_nodes()):
                l = self.graph.get_node(self.graph.node_ids[i]).labels[label_ind]
                if l in [0,1]:
                    out.write('obs ImageLabel(Image[{0}]) = {1};\n'.format(i, float(l)))
            out.write('\n')

            for i in range(self.graph.num_nodes()):
                l = self.graph.get_node(self.graph.node_ids[i]).labels[label_ind]
                if l not in [0,1]:
                    out.write('query ImageLabel(Image[{0}]);\n'.format(i))
            out.write('\n')


if __name__ == "__main__":
    #arguments: input_dir output_dir nlabels
    d = Data()
    d.add_nodes(os.path.join(sys.argv[1], 'testing', 'image_table.txt'), os.path.join(sys.argv[1], 'testing', 'caffe_histograms.txt'))
    d.add_edges(os.path.join(sys.argv[1], 'testing', 'image_edge_table.txt'))

    d.build_matrices()
    d.save_matrices(sys.argv[2])
    for i in range(int(sys.argv[3])):
        d.print_eval_code(i, 'cp6_eval.blog', os.path.join(sys.argv[2], 'eval_cp6_{0}.blog'.format(i)))
