import pickle
import os.path as osp
import torch_geometric
import networkx as nx
import os
import glob
import yaml
import csv
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
import re
from sklearn.preprocessing import OneHotEncoder
from worde4mde import load_embeddings

import pandas as pd

from pyparsing import col

from torch_geometric.data import HeteroData
edge_types = []



class Node:
  def __init__(self, filename):
    self.data, self.attributes = self.load_data(filename)

  def load_data(self, filename):
    df = pd.read_csv(filename)
    filename = Path(filename).name
    node_type = filename.rsplit("_", 1)[0]
    src = node_type.split("_")[0]
    id_col = f"{src}Id"
    attributes = df[list(df.columns)[1:]]
    data = df[[id_col, 'graphNumber']]
    return data, attributes

class Edge:
  def __init__(self, filename):
    self.data, self.attributes = self.load_data(filename)

  def load_data(self, filename):
      df = pd.read_csv(filename)
      filename = Path(filename).name
      edge_type = filename.rsplit("_", 1)[0]

      # edge_type = Path(filename).rsplit("_", 1)[0]

      m = re.match(r"^(.*)_(.*)$", edge_type)
      if m:
          parts = m.groups()
      else:
          raise ValueError(...)

      src = parts[0]
      dst = parts[1]

      src_col = f"sr{src}Id"
      dst_col = f"ds{dst}Id"
      attributes = list(df.columns)[2:]
      data = df[[src_col, dst_col]]
      return data, attributes
class CreateGraph:
    def __init__(self, data_path, filters):
        self.nodes = {}
        self.edges = {}
        self.filters = filters
        self.metamodel_config = []


        node_files = self.get_node_files(data_path)
        for file in node_files:
            filename = Path(file).name
            node_type = filename.rsplit("_", 1)[0]

            # edge_type = file?name.rsplit("_", 1)[0]
            self.nodes[node_type] = Node(file)

        edge_files = self.get_edge_files(data_path)
        for file in edge_files:
            filename = Path(file).name
            edge_type = filename.rsplit("_", 1)[0]
            self.edges[edge_type] = Edge(file)
        self.edit_data()


    #edit nodes and edges data according ti Yaml file
    def edit_data(self):
        nodes_copy = self.nodes.copy()
        edges_copy = self.edges.copy()


        for node_type, node in nodes_copy.items():
            include = self.filters.get("classes").get("include")

            exclude = self.filters.get("classes").get("exclude")
            if (include is not None and node_type not in include) or (exclude is not None and node in exclude):
                del self.nodes[node_type]

            # include or exclude the attributes of classes
            # By Default, the attributes of nodes are not considered
            # But they are considered if includeAllAttributes = true

            if self.filters.get("classes").get("includeAllAttributes") == False or self.filters.get("classes").get("excludeAllAttributes") == True:
                if self.nodes.get(node_type) is not None:
                    self.nodes[node_type].attributes = []

        for edge in edges_copy:
            # print("edges", edge)
            edge_type_parts = edge.split("_")
            source_node=edge_type_parts[0]
            destination_node = edge_type_parts[1]
            if include is not None and (source_node and destination_node) not in include:
                del self.edges[edge]
            if exclude is not None and (source_node or destination_node) in exclude:
                del self.edges[edge]


    def get_node_files(self, data_path):
        return glob.glob(f"{data_path}/nodes/*.csv")

    def get_edge_files(self, data_path):
        return glob.glob(f"{data_path}/edges/*.csv")

    def get_edge_type(self, filename):
        return Path(filename).stem

    def onehot_encode(self, encoding, col, value):
        unique = list(encoding[col]["unique_values"])
        num_rows = len(unique)

        # Reshape based on number of rows
        unique = np.reshape(unique, (num_rows, 1))

        encoder = OneHotEncoder()
        encoder.fit(unique)
        val = np.array(value)

        val = val.reshape(1, -1)
        new_val = encoder.transform(val).toarray()
        return new_val

    def word2vec_encode(self, encoding, col, value):
        unique_names = encoding[col]["unique_values"]
        names = [[word] for word in unique_names]
        modelnamestates = Word2Vec(names, min_count=1)
        modelnamestates.save('word2vec.modelnamestate')
        import gensim
        modelnamestates = gensim.models.Word2Vec.load('word2vec.modelnamestate')
        all_embeddings_namestates = {}
        for word in unique_names:
            if word in modelnamestates.wv:
                embedding = modelnamestates.wv[word]
                all_embeddings_namestates[word] = embedding

        new_val = all_embeddings_namestates[value]
        return new_val

    def worde4mde (self, value):
        sgram_mde = load_embeddings('sgram-mde')
        new_val = sgram_mde[value]
        return new_val

    def model2heterograph(self) -> HeteroData:
        graphs = {}
        # data = HeteroData()

        for node_type, node in self.nodes.items():
            encoding = {}
            # unique_values = set()
            columns = list(node.attributes)
            for column in columns:
                if self.filters.get('classes', {}).get(node_type, {}).get('features', {}).get(column, {}).get(
                        'encoding') is not None:
                    encoding[column] = {}
                    encoding[column]["enc_type"] = self.filters.get('classes', {}).get(node_type, {}).get('features',
                                                                                                          {}).get(
                        column, {}).get(
                        'encoding')
                    unique_values = set(node.attributes[column])
                    encoding[column]["unique_values"] = unique_values
            for _, row in node.data.iterrows():
                graphNum = row["graphNumber"]
                if graphs.get(graphNum) is None:
                    G = HeteroData()
                    graphs[graphNum] = G
                else:
                    G = graphs[graphNum]

                id = node_type.split("_")[0]
                attr_dict = {}

                for col in node.attributes.columns:
                    for val in node.attributes[col]:
                        if encoding.get(col) is not None:
                            if encoding[col]["enc_type"] == "one-hot":
                                new_val = self.onehot_encode(encoding,col, val)
                            if encoding[col]["enc_type"] == "word2vec":
                                new_val= self.word2vec_encode(encoding, col, val)
                            if encoding[col]["enc_type"] == "worde4mde":
                                new_val = self.worde4mde(val)
                            val = new_val
                        attr_dict[col] = val
                G[node_type].x = attr_dict

        for edge_type, edge in self.edges.items():
            for _, row in edge.data.iterrows():
                src = edge_type.split("_")[0]
                dst = edge_type.split("_")[1]
                G[edge_type].edge_index = self.get_edge_index(edge.data, src, dst)
                G[edge_type].attrs = edge.attributes

        for edge_type, edge in self.edges.items():
            src = edge_type.split("_")[0]
            dst = edge_type.split("_")[1]

            G[edge_type].edge_index = self.get_edge_index(edge.data, src, dst)
            # data[edge_type].edge_index = self.get_edge_index(edge.data)
            G[edge_type].attrs = edge.attributes


        return graphs

    def get_edge_index(self, edge_data, src, dst):
        src_col = f"sr{src}Id"
        dst_col = f"ds{dst}Id"
        return edge_data[[src_col, dst_col]].values

    def model2HomoGraph(self) -> nx.Graph:
        graphs = {}
        ids = {}


        # Add nodes
        for node_type, node in self.nodes.items():
            encoding = {}
            columns = list(node.attributes)
            for column in columns:
                if self.filters.get('classes', {}).get(node_type, {}).get('features', {}).get(column, {}).get(
                    'encoding') is not None:
                    encoding[column] = {}
                    encoding[column]["enc_type"] = self.filters.get('classes', {}).get(node_type, {}).get('features', {}).get(column, {}).get(
                    'encoding')
                    unique_values = set(node.attributes[column])
                    encoding[column]["unique_values"] = unique_values

            index = 0
            for _, row in node.data.iterrows():
                graphNum = row["graphNumber"]
                # Create graph if not exists
                if graphs.get(graphNum) is None:
                    G = nx.MultiDiGraph()
                    graphs[graphNum] = G
                    ids[graphNum] =0

                    # Create new graph
                    # Add to graphs list
                else:
                    # Get existing graph

                    G = graphs[graphNum]
                id = node_type.split("_")[0]
                nid = row[f"{id}Id"]
                node.data.loc[index, 'newId'] = ids[graphNum]
                index += 1
                newId = ids[graphNum]
                ids[graphNum] += 1

                G.add_node(newId)
                G.nodes[newId]['type'] = node_type
                for col in node.attributes.columns:
                    for val in node.attributes[col]:
                        if encoding.get(col) is not None:
                            if encoding[col]["enc_type"] == "one-hot":
                                new_val = self.onehot_encode(encoding,col, val)
                            if encoding[col]["enc_type"] == "word2vec":
                                new_val= self.word2vec_encode(encoding, col, val)
                            if encoding[col]["enc_type"] == "worde4mde":
                                new_val = self.worde4mde(val)

                                # new_val = new_val.reshape(1, -1)
                            val = new_val

                        G.nodes[newId][col] = val


        # Add edges
        for edge_type, edge in self.edges.items():

            for _, row in edge.data.iterrows():
                part1 = edge_type.split("_")[0]
                part2 = edge_type.split("_")[1]
                src = row[f"sr{part1}Id"]
                dst = row[f"ds{part2}Id"]

                newsrc = int(self.nodes[part1].data.loc[self.nodes[part1].data[f"{part1}Id"] == src, 'newId'].iloc[0])
                newdst = int(self.nodes[part2].data.loc[self.nodes[part2].data[f"{part2}Id"] == dst, 'newId'].iloc[0])

                graph_number =  self.nodes[part1].data.loc[self.nodes[part1].data[f"{part1}Id"] == src, 'graphNumber'].iloc[0]
                print("before:")
                print("g nodes:", G.nodes)
                graphs[graph_number].add_edge(newsrc, newdst)


        return graphs
