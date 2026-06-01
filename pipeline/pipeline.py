#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:28:33 2025

@author: polo
"""

import os
import sys 
import torch
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from time import time
from pathlib import Path

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../Communication")

import roadscene2vec
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.data.dataset import SceneGraphDataset
from roadscene2vec.scene_graph.extraction import image_extractor as RealEx
from roadscene2vec.learning.util.scenegraph_trainer import Scenegraph_Trainer
from Communication.e2emodel import MIMOE2EModel
from sgautoencoder.sg_autoencoder import sg_autoencoder

sys.modules['util'] = roadscene2vec.util

class GBSED:
    """
    The complete pipeline, from extracting road scenes to the risk assessment, including graph compression and decompression.
    """
    def __init__(self, config:configuration, com_model_endpoint, 
                 batch_size:int=16, ebno=tf.constant(20, tf.float32)):
        self.config = config
        self.data = self._load_()
        self.batch_size = batch_size
        self.ebno = ebno
        self.sg_ae = sg_autoencoder(self.config)
        self._load_communicator_(com_model_endpoint)
        
    def _set_ebno(self, ebno):
        self.ebno = ebno
        
    def _load_communicator_(self, endpoint:str):
        """
        load the communicator endpoint""

        Parameters
        ----------
        endpoint : str
            Path to the trained communicator weights.

        Returns
        -------
        None.

        """
        self.model = MIMOE2EModel()
        self.model(1, tf.constant(0.0, tf.float32))
        self.model_weights_path = endpoint
        with open(self.model_weights_path, 'rb') as f:
            weights = pkl.load(f)
        
        for i, w in enumerate(weights):
            self.model.neural_rx.weights[i].assign(w)
        
    
    def _load_(self) -> SceneGraphDataset:
        """
        Extract scene graphs from the images folder and return the scenegraph dataset containing all the extracted scenes.

        Returns
        -------
        SceneGraphDataset
            The Dataset containing all the scene graphs extracted from the images folder.

        """
        
        scengraph_dataset = SceneGraphDataset()
        scengraph_dataset.dataset_save_path = self.config.location_data['input_path']
        
        if self.config.loading_type == "pickle":
            return scengraph_dataset.load()
        elif self.config.loading_type == "folder":
            sg_extraction_object = RealEx.RealExtractor(self.config)
            sg_extraction_object.load()
            return sg_extraction_object.getDataSet()
        
        
    def _format_storage_(self, labels, feature_nodes, L, comp_T):
        """
        Reformat the given arguments into a 1D array-like for data transmission.

        Parameters
        ----------
        labels : int array-like
            The labels of the detected objects.
        feature_nodes : 2D-matrix
            The feature nodes matrix.
        L : array-like
            The selected relationships present in the road scene.
        comp_T : 2D-matrix
            The compressed adjacency matrix.

        Returns
        -------
        to_serialize : array-like
            a reformat of the given arguments into a 1D array-like.

        """
        to_serialize = []
        to_serialize.append(len(labels))
        to_serialize.extend(labels)
        to_serialize.append(feature_nodes.size)
        to_serialize.extend(feature_nodes.ravel())
        to_serialize.append(L.size)
        to_serialize.extend(L)
        to_serialize.append(comp_T.size)
        to_serialize.extend(comp_T.ravel())
        to_serialize = np.asarray(to_serialize, dtype=np.float16)
        return to_serialize
    
    def _format_loading_(self, to_read):
        """
        From an 1D array, reads and returns the labels, features_nodes matrix, the selected indexes and the compressed adjacency matrix.

        Parameters
        ----------
        to_read : 1D array-like
            DESCRIPTION.

        Returns
        -------
        labels : int array-like
            The labels of the detected objects.
        feature_nodes : 2D-matrix
            The feature nodes matrix.
        L : array-like
            The selected relationships present in the road scene.
        comp_T : 2D-matrix
            The compressed adjacency matrix.

        """
        cur_idx, end_idx, nb = 0, 0, 0
        
        # Labels reading
        nb = int(to_read[cur_idx])
        cur_idx += 1
        end_idx = cur_idx + nb
        labels = to_read[cur_idx : end_idx]
        
        # Features reading
        cur_idx = end_idx
        nb = int(to_read[cur_idx])
        cur_idx += 1
        end_idx = cur_idx + nb
        features = to_read[cur_idx : end_idx]
        
        # Indices reading
        cur_idx = end_idx
        nb = int(to_read[cur_idx])
        cur_idx += 1
        end_idx = cur_idx + nb
        L = to_read[cur_idx : end_idx]
        
        # Compressed adj matrices reading
        cur_idx = end_idx
        nb = int(to_read[cur_idx])
        cur_idx += 1
        end_idx = cur_idx + nb
        comp_T = to_read[cur_idx : ]
        
        # Reorganization of all read arrays
        labels = [int(i) for i in labels]
        feature_nodes = features.reshape(((len(labels)), -1))
        L = [int(i) for i in L]
        comp_T = comp_T.reshape((len(L), len(labels), len(labels)))
        
        return labels, feature_nodes, L, comp_T
        
    def _prepare_bits_for_model_(self, input_bits_1d):
        
        if not (isinstance(input_bits_1d, np.ndarray) and input_bits_1d.ndim == 1) and \
           not (isinstance(input_bits_1d, tf.Tensor) and input_bits_1d.shape.rank == 1):
            raise ValueError("Input bits must be a 1D numpy array or TensorFlow tensor.")
    
        try:
            model_k = self.model.k.numpy()
        except AttributeError:
            model_k = self.model.k
        except Exception as e:
            print(f"Could not get `model.k` from the model object. Error: {e}")
            return None
            
        expected_bits = self.batch_size * self.model.num_ut * self.model.num_tx_ant * model_k
    
        if len(input_bits_1d) > expected_bits:
            prepared_bits = input_bits_1d[:expected_bits]
        elif len(input_bits_1d) < expected_bits:
            padding_size = expected_bits - len(input_bits_1d)
            prepared_bits = np.pad(input_bits_1d, (0, padding_size), 'constant', constant_values=0)
        else:
            prepared_bits = input_bits_1d
    
        return tf.constant(prepared_bits, dtype=tf.int32)
    
    def _recover_original_bits_(self, decoded_bits, original_length):
        decoded_bits_flat = tf.reshape(decoded_bits, [-1])
        recovered_bits = decoded_bits_flat[:original_length]
        return recovered_bits
    
    def _to_bits_array_(self, np_array):
        b = np_array.tobytes()
        bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
        return bits

    def _to_float_array_(self, bits):
        b = np.packbits(bits)
        float_array = np.frombuffer(b.tobytes(), np.float16)
        return float_array
    
    def _process_sg_(self, sg):
        """
        _process_sg_ extracts the nodes labels, the features_node_matrix and the adjacency matrices tensor from the SceneGraph sg.

        Parameters
        ----------
        sg : SceneGraph
            The intermediate representation of a raod scene image.

        Returns
        -------
        dict
            All the relevant information extracted from the SceneGraph object sg.

        """
        labels, feat_nodes_mat, T = self.sg_ae.encode(sg)
        comp_T, L = self.sg_ae.sem_compression(T)
        to_serialize = self._format_storage_(labels, feat_nodes_mat, L, comp_T)
        input_bits = self._to_bits_array_(to_serialize)
        return {
            "labels": labels, 
            "feature_nodes_matrix":feat_nodes_mat, 
            "compressed_Tensor":comp_T, 
            "indexes":L, 
            "input_bits":input_bits, 
            "prepared_bits":self._prepare_bits_for_model_(input_bits)
        }
    
    def _sg_reconstruction_(self, received_bits, initial_processed_sg):
        """
        _sg_reconstruction_ reconstructs SceneGraph object by proceeding the transmitted data. 
        Compare the reconstructed SceneGraph to the initial SceneGraph to insure a consistant information transmission.

        Parameters
        ----------
        received_bits : np.ndarray
            The received bits after the data transmission.
        initial_processed_sg : dict
            The information about the initial SceneGraph.

        Returns
        -------
        rec_sg : SceneGraph
            The reconstructed SceneGraph from the received bits.

        """
        to_read = self._to_float_array_(received_bits)
        labels = initial_processed_sg['labels']
        comp_T = initial_processed_sg['compressed_Tensor']
        feat_nodes_mat = initial_processed_sg['feature_nodes_matrix']
        L = initial_processed_sg['indexes']
        try: 
            rec_labels, rec_feature_nodes, rec_L, rec_comp_T = self._format_loading_(to_read)
            if np.allclose(comp_T, rec_comp_T) \
                and np.allclose(labels, rec_labels) \
                and np.allclose(feat_nodes_mat, rec_feature_nodes) \
                and np.allclose(L, rec_L):
                rec_sg = self.sg_ae.decode(rec_labels, rec_feature_nodes, rec_L, rec_comp_T)
                return rec_sg
        except Exception:
            print("\nTruncated file", file=sys.stderr)

    
    def sg_transmission(self, sg):
        """
        sg_transmission simulates the data transmission of the processed SceneGraph object sg. 
        Measures the transmission time and returns the reconstructed SceneGraph from the received data, and the transmission time.

        Parameters
        ----------
        sg : SceneGraph
            The SceneGraph to process and transmit.

        Returns
        -------
        tuple
            (The reconstructed SceneGraph, the transmission time).

        """
        initial_processed_sg = self._process_sg_(sg)
        start = time()
        b, b_hat = self.model(
            self.batch_size, 
            self.ebno, 
            initial_processed_sg['prepared_bits']
        )
        end = time()
        received_bits = self._recover_original_bits_(
            b_hat, 
            initial_processed_sg['input_bits'].size
        ).cpu().numpy()
        received_bits = np.asarray(received_bits, dtype=np.uint8)
        return self._sg_reconstruction_(received_bits, initial_processed_sg), (end - start)
        
    def e2e(self):
        path = self.config.location_data['input_path']
        if self.config.loading_type == "pickle":
            location = self.data.dataset_path
            if not os.path.exists(location):
                os.mkdir(location)
        elif self.config.loading_type == "folder":
            location = path
            
        scene_graphs = self.data.scene_graphs
        folder_names = self.data.folder_names
        sent_labels = self.data.labels
        received_scene_graphs, received_labels, received_folder_names = {}, {}, []
        correctly_transmitted, total_file_nb = 0, 0
        keys = list(scene_graphs.keys())
        durations = []
        
        p_bar = tqdm(range(len(keys)))
        for i in p_bar:
            key = keys[i]
            p_bar.set_description(f"folder={key}, ebno={ebno}")
            folder_name = folder_names[i]
            seq_folder = os.path.join(location, folder_name)
            if not os.path.exists(seq_folder):
                os.mkdir(seq_folder)
            to_store_folder = os.path.join(seq_folder, "encoded_files")
            if not os.path.exists(to_store_folder):
                os.mkdir(to_store_folder)
            received_folder = os.path.join(seq_folder, "received_files")
            if not os.path.exists(received_folder):
                os.mkdir(received_folder)
            sequence = scene_graphs[key]
            received_scene_graphs[key] = {}
            for seq_file_num in sequence.keys():
                sg = sequence[seq_file_num]
                rec_sg, duration = self.sg_transmission(sg)
                sg.visualize(os.path.join(to_store_folder, str(seq_file_num) + ".png"))
                if not rec_sg is None:
                    received_scene_graphs[key][seq_file_num] = rec_sg
                    rec_sg.visualize(os.path.join(received_folder, str(seq_file_num) + ".png"))
                    correctly_transmitted += 1
                durations.append(duration)
            
            total_file_nb += len(sequence)
            
            if len(received_scene_graphs[key]) > 0:
                received_labels[key] = sent_labels[key]
                received_folder_names.append(folder_name)
            else:
                received_scene_graphs.pop(key)   
                
            scene_graph_dataset = SceneGraphDataset()
            scene_graph_dataset.scene_graphs = received_scene_graphs
            scene_graph_dataset.labels = received_labels
            scene_graph_dataset.folder_names = received_folder_names
            scene_graph_dataset.dataset_save_path = self.config.location_data['data_save_path']
            scene_graph_dataset.dataset_type = self.config.dataset_type
            
        return scene_graph_dataset, correctly_transmitted, total_file_nb, durations
    
def main(learning_filename, pipe, time_file):    
    sg_dataset, c_transmitted, total_file_nb, durations = pipe.e2e()
    if len(sg_dataset.scene_graphs) > 0 \
        and 1.0 in list(sg_dataset.labels.values()):
        sg_dataset.save()
        learning_config = configuration(learning_filename, from_function=True)
        trainer = Scenegraph_Trainer(learning_config)
        trainer.build_transfer_learning_dataset()
        trainer.build_model()
        trainer.load_model()
        ret_values = trainer.inference(trainer.transfer_data, trainer.transfer_data_labels)
        outputs, labels = ret_values[0], ret_values[1]
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        df = pd.DataFrame({f"duration_{int(ebno)}":durations})
        if not os.path.exists(time_file):
            df.to_csv(time_file, index=False)
        else: 
            df1 = pd.read_csv(time_file)
            df2 = pd.concat((df, df1), axis=1)
            df2.to_csv(time_file, index=False)
            
        return labels, preds, c_transmitted, total_file_nb
    

def main_parser():
    parser = argparse.ArgumentParser(
        description="The complete pipeline from scene graph extraction"\
            "semantic enconding, wireless communication, semantic decoding, "\
            "to risk assessment or collision prediction."
    )
    parser.add_argument(
        "--extraction_filename", 
        type=str, 
        default="../Config/pipeline_extraction.yaml", 
        help="Path to scene graph extraction configuration file."
    )
                                     
    parser.add_argument(
        "--learning_filename", 
        type=str,
        default="../Config/pipeline_learning.yaml", 
        help="Path to the model learning configuration file."
    )
    parser.add_argument(
        "--com_model_endpoint", 
        type=str, 
        default="../Communication/weights/neural_rx_ofdm_mimo_cdl_final.h5", 
        help="Path to the communication model checkpoint."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="../../Data/Outputs/outputs.csv", 
        help="Path to save the outputs"
    )
    parser.add_argument(
        "--sem_fid_file", 
        type=str, 
        default="../../Data/Outputs/sem_fidelity.csv", 
        help="Path to save the semantic fidelity dataframe"
    )
    parser.add_argument(
        "--time_file", 
        type=str, 
        default="../../Data/Outputs/transfert_times.csv", 
        help="Path to save the transfert times dataframe"
    )
    return parser.parse_args()
        
            
if __name__ == "__main__":
    args = main_parser()
    extraction_config = configuration(
        args.extraction_filename, 
        from_function=True
    )
    output_parent = Path(args.output_file).parent
    if not os.path.exists(output_parent):
        os.makedirs(output_parent)
        print('Created directory : %s' % output_parent)
    df, sem_fid_df = pd.DataFrame(), pd.DataFrame()
    pipe = GBSED(extraction_config, args.com_model_endpoint)
    for e in np.linspace(0, 20, 11):
        ebno = tf.constant(e, tf.float32)
        pipe._set_ebno(ebno)
        for i in range(1):
            main_return = main(args.learning_filename, pipe, args.time_file)
            if not main_return is None: 
                labels, preds, correctly_transmitted, total_file_nb = main_return
                df = pd.concat(
                        (df, 
                         pd.DataFrame(
                            {
                                "iteration": i * np.ones(len(labels)),
                                "label":labels, 
                                 "prediction":preds,  
                                 "ebno":[e for i in range(len(labels))]
                             })
                        ), 
                        axis=0
                    )
                sem_fid_df = pd.concat(
                    (
                        sem_fid_df, 
                         pd.DataFrame(
                             {
                                 "iteration":[i], 
                                 "ebno" : [e], 
                                 "correct_transmission":[correctly_transmitted], 
                                 "total_files":[total_file_nb]
                             }
                        )
                    ), 
                    axis=0
                )
                df.to_csv(args.output_file, index=False)
                sem_fid_df.to_csv(args.sem_fid_file, index=False)