#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:12:39 2025

@author: polo
"""

import os
import numpy as np
import torch
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from learning.rs2vec_training import rs_trainer
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.scene_graph.scene_graph import SceneGraph

class sg2text:
    def __init__(self, config:configuration):
        super(sg2text, self ).__init__()
        self.relation_map = config.relation_map
    
    def scene_graph_to_prompt(self, sg:SceneGraph)->str:
        prompts = []
        adjacency_matrix = sg.g.adjacency()
        for subject, edges in adjacency_matrix:
            if "lane" in subject.name.lower() or not edges:
                continue

            readable_subject = sg2text.make_readable(subject.name)

            for target, rels in edges.items():
                rel_labels = [v['label'] for v in rels.values()]
                if subject == target:
                    continue  # skip self-relations

                readable_target = sg2text.make_readable(target.name)

                # Handle "is in" as a special case
                if 'isIn' in rel_labels:
                    prompts.append(f"{readable_subject} is in {readable_target}.")
                    continue

                # Convert relation types to readable text
                readable_rels = ', '.join([self.relation_map.get(rel) for rel in rel_labels])

                prompts.append(f"{readable_subject} is {readable_rels} {readable_target}.")
        descr_list = [p.capitalize() for p in prompts]
        return " ".join(descr_list)
            
    @staticmethod
    def make_readable(node_name):
        if "_" in node_name:
            return " ".join(["the"] + node_name.split("_")[::(-1 if "lane" in node_name else 1)])
        return " ".join(["the", node_name])

class SgSdDataset(Dataset):
    def __init__(
            self, 
            scene_graphs, 
            images_path, 
            config, 
            transform_img=None, 
            tokenizer=None
            ):
        super(SgSdDataset, self).__init__()
        assert len(scene_graphs) == len(images_path), "Scene_graphs and images_path must be the same size"
        self.scene_graphs = scene_graphs
        self.images_path = images_path
        self.config = config
        self.descriptor = sg2text(self.config)
        self.transform_img = transform_img
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.scene_graphs)
    
    def __getitem__(self, index:int):
        sg = self.scene_graphs[index]
        caption = self.descriptor.scene_graph_to_prompt(sg)
        img = Image.open(self.images_path[index]).convert("RGB")
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        for node, data in sg.g.nodes(data=True):
            if node.name.startswith("car_"):
                left = int(data["attr"]["left"])
                top = int(data["attr"]["top"])
                right = int(data["attr"]["right"])
                bottom = int(data["attr"]["bottom"])
                mask 
                mask[top:bottom, left:right] = 1
        boxes = torch.from_numpy(mask).unsqueeze(0)
        if not self.transform_img is None:
            img = self.transform_img(img)
        if not self.tokenizer is None:
            token = self.tokenizer(
                caption, 
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length
            )
            input_ids = token.input_ids.squeeze(0)
            attention_mask = token.attention_mask.squeeze(0)
        return caption, img, boxes, input_ids, attention_mask

class ds_gen:
    def __init__(
            self,
            config_filename, 
            input_datasets, 
            transform_img=None, 
            tokenizer=None,
        ):
        self.train_sg_dataset_path = input_datasets["train_dataset"]
        self.valid_sg_dataset_path = input_datasets['valid_dataset']
        self.config_filename = config_filename
        self.config = configuration(self.config_filename, from_function=True)
        self.tokenizer = tokenizer
        self.transform_img = transform_img
        self.valid_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        
    def load(self):
        def get_sg_img_path(dataset):
            ds, img_paths = [], []
            location = dataset.dataset_path
            scene_graphs = dataset.scene_graphs
            keys = list(scene_graphs.keys())
            for i in range(len(scene_graphs)):
                key = keys[i]
                folder = dataset.folder_names[i]
                path = os.path.join(location, folder, 'raw_images')
                images = os.listdir(path)
                for frame in scene_graphs[key].keys():
                    sg = scene_graphs[key][frame]
                    image = [i for i in images if i.split('.')[0].endswith(str(frame))][0]
                    ds.append(sg)
                    img_paths.append(os.path.join(path, image))
            return ds, img_paths
                    
        if not os.path.exists(self.train_sg_dataset_path) \
            or not os.path.exists(self.valid_sg_dataset_path):
                r_trainer = rs_trainer(self.config)
                r_trainer.load_datasets()
        with open(self.train_sg_dataset_path, "rb") as f: 
            train_sg_dataset = pkl.load(f)
        with open(self.valid_sg_dataset_path, 'rb') as f:
            valid_sg_dataset = pkl.load(f)
        train_ds, train_images = get_sg_img_path(train_sg_dataset)
        val_ds, val_images = get_sg_img_path(valid_sg_dataset)
        return train_ds, train_images, val_ds, val_images
                
    def create_datasets(self, test_size=0.30): 
        train_sg, train_images, val_sg, val_images = self.load()
        assert len(train_sg) == len(train_images) \
            and  len(val_sg) == len(val_images)
        train_indices, test_indices = train_test_split(
            list(range(len(train_sg))), 
            test_size=test_size, 
            random_state=42
        )
        train_scene_graphs, train_images_path = [], []
        for i in train_indices:
            train_scene_graphs.append(train_sg[i])
            train_images_path.append(train_images[i])
        test_scene_graphs, test_images_path = [], []
        for i in test_indices:
            test_scene_graphs.append(train_sg[i])
            test_images_path.append(train_images[i])
        self.train_dataset = SgSdDataset(
            train_scene_graphs, 
            train_images_path, 
            self.config, 
            self.transform_img, 
            self.tokenizer
        )
        self.test_dataset = SgSdDataset(
            test_scene_graphs, 
            test_images_path, 
            self.config, 
            self.transform_img, 
            self.tokenizer
        )
        self.valid_dataset = SgSdDataset(
            val_sg, 
            val_images, 
            self.config, 
            self.transform_img, 
            self.tokenizer
        )
        
    def save_datasets(self, train_path, test_path, valid_path):
        paths = [train_path, test_path, valid_path]
        if self.train_dataset is None \
            or self.test_dataset is None \
            or self.valid_dataset is None:
            self.create_datasets()
        datasets = [self.train_dataset, self.test_dataset, self.valid_dataset]
        for path, data in zip(paths, datasets):
            with open(path, "wb") as f:
                pkl.dump(data, f)
        print("Datasets saved !")