#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:51:34 2025

@author: polo
"""

import os
import sys
import wandb
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef,\
    precision_score, recall_score, roc_auc_score

sys.path.append('../..')

import roadscene2vec
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.data.dataset import SceneGraphDataset
from roadscene2vec.scene_graph.extraction import image_extractor as RealEx
from roadscene2vec.learning.util.scenegraph_trainer import Scenegraph_Trainer

sys.modules['util'] = roadscene2vec.util


class rs_trainer:
    """
        Class implementing a trainer for road scene data
    """
    
    def __init__(
            self, 
            train_config_filename:str=None, 
            train_learning_config_filename:str=None):
        
        self.training_config = configuration(
            train_config_filename, 
            from_function=True
        )
        self.learning_config = configuration(
            train_learning_config_filename,
            from_function=True
        )            
    
    
    def load_datasets(self):
        """
        If the training and validation datasets exist, it loads them into memory;
            otherwise, it creates them and then loads them into memory.

        Returns
        -------
        None.

        """
        train_data_save_path = self.training_config.location_data['data_save_path']
        val_data_save_path = train_data_save_path[:-9] + "valid.pkl"
        if not os.path.exists(train_data_save_path) \
            or not os.path.exists(val_data_save_path):
            extraction_obj = RealEx.RealExtractor(self.training_config)
            extraction_obj.load()
            dataset = extraction_obj.getDataSet()
            scene_graphs = dataset.scene_graphs
            keys = list(scene_graphs.keys())
            X_train, X_val, _, _ = train_test_split(
                list(range(len(keys))), 
                list(dataset.labels.values()),
                test_size=0.10, 
                random_state=17
            )
            train_data_save_path = self.training_config.location_data[
                "data_save_path"
            ]
            train_dataset = SceneGraphDataset()
            val_dataset = SceneGraphDataset()
            train_dataset.dataset_path = self.training_config.location_data['input_path']
            train_dataset.dataset_save_path = train_data_save_path
            val_dataset.dataset_path = self.training_config.location_data['input_path']
            val_dataset.dataset_save_path = val_data_save_path
            val_scene_graphs, val_labels, val_folder_names = {}, {}, []
            for i in X_val:
                key = keys[i]
                val_scene_graphs[key] = scene_graphs[key]
                val_labels[key] = dataset.labels[key]
                val_folder_names.append(dataset.folder_names[i])
            val_dataset.scene_graphs = val_scene_graphs
            val_dataset.folder_names = val_folder_names
            val_dataset.labels = val_labels
            train_scene_graphs, train_labels, train_folder_names = {}, {}, []
            for i in X_train:
                key = keys[i]
                train_scene_graphs[key] = scene_graphs[key]
                train_labels[key] = dataset.labels[key]
                train_folder_names.append(dataset.folder_names[i])
            train_dataset.scene_graphs = train_scene_graphs
            train_dataset.folder_names = train_folder_names
            train_dataset.labels = train_labels
            self.train_dataset = train_dataset                
            self.valid_dataset = val_dataset
            self.train_dataset.save()
            self.valid_dataset.save()
        else: 
            sg_dataset = SceneGraphDataset()
            sg_dataset.dataset_save_path = train_data_save_path
            self.train_dataset = sg_dataset.load()
            sg_dataset = SceneGraphDataset()
            sg_dataset.dataset_save_path = val_data_save_path
            self.valid_dataset = sg_dataset.load()


    def train(self):
        """
        Train either the action classification model or the collision prediction 
            model according to the desired configuration.

        Returns
        -------
        None.

        """
        run = wandb.init(
            project=self.learning_config.wandb_configuration["project"], 
            entity=self.learning_config.wandb_configuration['entity']
        )
        trainer = Scenegraph_Trainer(self.learning_config, run)
        trainer.split_dataset()
        trainer.build_model()
        trainer.learn()
        run.finish()
        num_par = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print("Number of parameters : %d" %num_par)
        
    def evaluate(self): 
        """
        Evaluate the trained model on the validation data set. 

        Returns
        -------
        None.

        """
        trainer = Scenegraph_Trainer(self.learning_config)
        trainer.build_transfer_learning_dataset()
        trainer.build_model()
        trainer.load_model()
        ret = trainer.inference(trainer.transfer_data, 
                                trainer.transfer_data_labels)
        outputs = ret[0]
        labels = ret[1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Metrics
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        roc = roc_auc_score(labels, preds)
        
        print("Accuracy score : %.3f" %acc)
        print("Precision : %.3f" %prec)
        print("Recall : %.3f" %recall)
        print("F1-score : %.3f" %f1)
        print("mcc : %.3f" %mcc)
        print("roc : %.3f" %roc)

if __name__ == "__main__":
    train_config_filename = '../Config/task_oriented_extraction_config.yaml'
    learning_config_filename = '../Config/task_oriented_learning_config.yaml'
    
    r_trainer = rs_trainer(
        train_config_filename, 
        learning_config_filename
    )
    r_trainer.load_datasets()
    r_trainer.train()
    r_trainer.evaluate()