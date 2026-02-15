#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:40:30 2025

@author: polo
"""

import sys
import numpy as np

sys.path.append("..")

from roadscene2vec.scene_graph.nodes import Node
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.scene_graph.scene_graph import SceneGraph
from roadscene2vec.scene_graph.extraction import image_extractor as RealEx


class sg_autoencoder(object):
    """
    sg_encoder : provides the tools to save scene-graphs as adjacency matrices 
        and serialize them in binary format. It also lets you reconstruct graphs 
        from binary files. 
    """

    def __init__(self, config:configuration):
        super(sg_autoencoder, self).__init__()
        self.config = config
        self.rels = self.config.relation_extraction_settings["RELATION_NAMES"]
        self.sg_extraction_object = RealEx.RealExtractor(self.config)
       
    def _get_node_label_idx(self, node):
        return self.config.relation_extraction_settings["ACTOR_NAMES"].index(node.label)
    
    def _get_all_node_names(self, labels, features):
        names = []
        for i in range(len(labels)):
            l_idx = labels[i]
            to_append = ""
            name = self.config.relation_extraction_settings["ACTOR_NAMES"][l_idx]
            if name == "ego_car":
                to_append = "ego car"
            elif name == "road":
                to_append = "Root Road"
            elif name == "lane":
                if "Right Lane" in names:
                    to_append = "Middle Lane"
                else: 
                    to_append = "Right Lane" if "Left Lane" in names else "Left Lane"
            else: 
                suffix = int(features[i, -1])
                to_append = "%s_%d" %(name, suffix)
            names.append(to_append)
        return names
       
    def _get_node(self, name, features):
        if "_" in name: 
            actor_type, _ = name.split("_")
            actor_value = self.config.relation_extraction_settings["ACTOR_NAMES"].index(actor_type)
            attr = {
                'left': features[0], 
                'top': features[1], 
                'right': features[2], 
                'bottom': features[3], 
                'location_x':features[4], 
                'location_y':features[5], 
                'rel_location_x':features[6], 
                'rel_location_y':features[7], 
                'distance_abs':features[8]
            }
            node = Node(name, attr, actor_type, actor_value)
        else: 
            if "road" in name.lower():
                node = Node(name, {'name' : "Root Road"}, "road", self.config.relation_extraction_settings["ACTOR_NAMES"].index("road"))
            elif "ego" in name.lower():
                node = Node(name, 
                            {"location_x":features[0], "location_y":features[1]}, 
                            'ego_car', self.config.relation_extraction_settings["ACTOR_NAMES"].index("ego_car"))
            elif "lane" in name.lower(): 
                node = Node(name, {}, "lane", self.config.relation_extraction_settings["ACTOR_NAMES"].index("lane"))
        return node
        
    def sem_compression(self, T):
        """
        Implements the semantic compression by removing the adjacency matrices 
            of relation that doesn't appear to the represented scene graph.

        Parameters
        ----------
        T : 3d-array
            Array of tensor of adjacency matrices of all of the relationship.

        Returns
        -------
        M : 3d-array
            The compressed tensor of adjacency matrices.
        L : 1d-array
            List of indexes of the relationship present in the scene graph.

        """
        M, L = [], []
        for i in range(len(self.rels)):
            if np.any(T[i] != 0): 
                M.append(T[i].copy())
                L.append(i)
        M = np.array(M, dtype=T.dtype)
        L = np.array(L, dtype=np.int8)
        return M, L    
    
    def sem_decompression(self, comp_T, L):
        """
        Implements the inverse operation of semantic compression.

        Parameters
        ----------
        comp_T : 3d-array
            The compressed tensor of adjacency matrices.
        L : 1d-array
            List of indexes of the relationship present in the scene graph.

        Returns
        -------
        T : 3d-array
            Array of tensor of adjacency matrices of all of the relationship.

        """
        shape = comp_T.shape[1:]
        T = np.zeros((len(self.rels), shape[0], shape[1]), dtype=comp_T.dtype)
        for i in range(len(self.rels)):
            if i in L: 
                pos = L.index(i)
                T[i] = comp_T[pos].copy() 
        return T    
    
    def encode(self, sg:SceneGraph):
        """
        Encode : encode a scene-graph as an adjacency matrix

        Parameters
        ----------
        sg : SceneGraph
            Objet representing the scene-graph of a road image.

        Returns
        -------
        M : 2D-ndarray 
            Adjacent matrix reprensenting sg.
        node_names : list of str
            The names of the graph's nodes.

        """
        adj_mat = dict(sg.g.adjacency())
        node_with_data = dict(sg.g.nodes(data=True))
        nodes = list(adj_mat.keys())
        labels, features = [], []
        n, m = len(self.rels), len(nodes)
        T = np.zeros((n, m, m), dtype=np.int8)
        for i_node in nodes:
            relations = adj_mat[i_node]
            labels.append(self._get_node_label_idx(i_node))
            features.append(list(node_with_data[i_node]["attr"].values()))
            i_node_idx = nodes.index(i_node)
            if relations:
                for f_node in relations.keys():
                    edges = relations[f_node]
                    f_node_idx = nodes.index(f_node)
                    for edge in edges.keys():
                        e = edges[edge]
                        T[e["value"], i_node_idx, f_node_idx] = 1
        feat_lengths = [len(f) for f in features]
        feat_nodes_mat = np.zeros(
            (len(nodes), max(feat_lengths) + 1), dtype=np.float16)
        for i in range(len(nodes)):
            feat = features[i]
            for j in range(len(feat)):
                if isinstance(feat[j], (float, np.float32)):
                    feat_nodes_mat[i, j] = feat[j]
            if "_" in nodes[i].name:
                suffix = int(nodes[i].name.split("_")[-1])
                feat_nodes_mat[i, -1] = suffix
        return labels, feat_nodes_mat, T
    
    def decode(self, labels, feature_nodes, indices, comp_T):
        T = self.sem_decompression(comp_T, indices)
        node_names = self._get_all_node_names(labels, feature_nodes)
        nodes = [self._get_node(node_names[i], feature_nodes[i])
                 for i in range(len(node_names))]
        sg = SceneGraph(self.sg_extraction_object.relation_extractor,
                        platform=self.config.dataset_type,
                        bev=self.sg_extraction_object.bev,
                        bounding_boxes=([], [], []))
        sg.g.clear()
        for node in nodes:
            sg.add_node(node)
        for i in range(len(self.rels)):
            for j in range(len(nodes)):
                for k in range(len(nodes)):
                    if T[i, j, k]:
                        i_node, f_node = nodes[j], nodes[k]
                        sg.add_relation((i_node, self.rels[i], f_node))
        return sg