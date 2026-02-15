#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:28:33 2025

@author: polo
"""

import os
import sys 
import torch
import numpy as np
import pickle as pkl
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../Communication/")
sys.path.append("../utils/")

import roadscene2vec
from utils.datasetGenerator import sg2text
from Communication.e2emodel import MIMOE2EModel
from sgautoencoder.sg_autoencoder import sg_autoencoder
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.data.dataset import SceneGraphDataset
from roadscene2vec.scene_graph.extraction import image_extractor as RealEx
from roadscene2vec.learning.util.scenegraph_trainer import Scenegraph_Trainer
sys.modules['util'] = roadscene2vec.util

class GBSED:
    def __init__(self, config:configuration, batch_size=16):
        self.config = config
        self.sg_ae = sg_autoencoder(self.config)
        self.data = self.__load__()
        self.batch_size = batch_size
        self.text_gen = sg2text(self.config)
        self.__load_communicator__()
        
    
    def __load_communicator__(self):
        self.model = MIMOE2EModel()
        self.model(1, tf.constant(0.0, tf.float32))
        self.model_weights_path = "../Communication/weights/neural_rx_ofdm_mimo_cdl_final.h5"
        with open(self.model_weights_path, 'rb') as f:
            weights = pkl.load(f)
        
        for i, w in enumerate(weights):
            self.model.neural_rx.weights[i].assign(w)
        
    
    def __load__(self) -> SceneGraphDataset:
        
        scengraph_dataset = SceneGraphDataset()
        scengraph_dataset.dataset_save_path = self.config.location_data['input_path']
        
        if self.config.loading_type == "pickle":
            return scengraph_dataset.load()
        elif self.config.loading_type == "folder":
            sg_extraction_object = RealEx.RealExtractor(self.config)
            sg_extraction_object.load()
            return sg_extraction_object.getDataSet()
        
        
    def _format_storage(self, labels, feature_nodes, L, comp_T):
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
    
    def _format_loading(self, to_read):
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
        
    def _prepare_bits_for_model(self, input_bits_1d):
        if not (isinstance(input_bits_1d, np.ndarray) and input_bits_1d.ndim == 1) and \
           not (isinstance(input_bits_1d, tf.Tensor) and input_bits_1d.shape.rank == 1):
            raise ValueError("Input bits must be a 1D numpy array or TensorFlow tensor.")
    
        # Get the number of bits per antenna from the model's internal configuration
        # The `k` attribute is a common way to get this in Sionna models.
        # We use .numpy() to convert the tf.Tensor k to a standard integer
        try:
            model_k = self.model.k.numpy()
        except AttributeError:
            # If the model.k attribute is not a tensor, it might be an integer directly
            model_k = self.model.k
        except Exception as e:
            print(f"Could not get `model.k` from the model object. Error: {e}")
            return None
            
        # Calculate the total number of bits the model expects
        expected_bits = self.batch_size * self.model.num_ut * self.model.num_tx_ant * model_k
    
        # Pad or truncate the input array to match the expected size
        if len(input_bits_1d) > expected_bits:
            # Truncate the array if it's too long
            prepared_bits = input_bits_1d[:expected_bits]
        elif len(input_bits_1d) < expected_bits:
            # Pad the array with zeros if it's too short
            padding_size = expected_bits - len(input_bits_1d)
            prepared_bits = np.pad(input_bits_1d, (0, padding_size), 'constant', constant_values=0)
        else:
            prepared_bits = input_bits_1d
    
        return tf.constant(prepared_bits, dtype=tf.int32)
    
    def _recover_original_bits(self, decoded_bits, original_length):
        # Flatten the decoded bits if they are not 1D
        # Sionna's output is often a multi-dimensional tensor (e.g., [batch, ut, ant, k])
        # The number of elements is what matters.
        decoded_bits_flat = tf.reshape(decoded_bits, [-1])
        
        # Slice the tensor to get the original number of bits
        recovered_bits = decoded_bits_flat[:original_length]
        
        return recovered_bits
    
    def _to_bits_array(self, np_array):
        b = np_array.tobytes()
        bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
        return bits

    def _to_float_array(self, bits):
        b = np.packbits(bits)
        float_array = np.frombuffer(b.tobytes(), np.float16)
        return float_array
        
    
    def e2e(self, ebno):
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
        p_bar = tqdm(range(len(keys)))
        for i in p_bar:
            key = keys[i]
            p_bar.set_postfix(
                folder=f"{key}", 
                ebno=f"{ebno}"
            )
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
            captions_folder = os.path.join(seq_folder, "captions")
            if not os.path.exists(captions_folder):
                os.mkdir(captions_folder)
            ebno_folder = os.path.join(captions_folder, str(int(ebno)))
            if not os.path.exists(ebno_folder):
                os.mkdir(ebno_folder)
            sequence = scene_graphs[key]
            received_scene_graphs[key] = {}
            for seq_file_num in sequence.keys(): 
                total_file_nb += 1
                sg = sequence[seq_file_num]
                labels, feat_nodes_mat, T = self.sg_ae.encode(sg)
                comp_T, L = self.sg_ae.sem_compression(T)
                to_serialize = self._format_storage(labels, feat_nodes_mat, L, comp_T)
                sg.visualize(os.path.join(to_store_folder, str(seq_file_num) + ".png"))
                input_bits = self._to_bits_array(to_serialize)
                prepared_bits = self._prepare_bits_for_model(input_bits)
                b, b_hat = self.model(self.batch_size, ebno, prepared_bits)
                received_bits = self._recover_original_bits(b_hat, input_bits.size).cpu().numpy()
                received_bits = np.asarray(received_bits, dtype=np.uint8)
                to_read = self._to_float_array(received_bits)
                # assert to_serialize.shape == to_read.shape, "%d, %d" %(key, seq_file_num)
                try: 
                    rec_labels, rec_feature_nodes, rec_L, rec_comp_T = self._format_loading(to_read)
                    if np.allclose(comp_T, rec_comp_T) \
                        and np.allclose(labels, rec_labels) \
                        and np.allclose(feat_nodes_mat, rec_feature_nodes) \
                        and np.allclose(L, rec_L):
                        rec_sg = self.sg_ae.decode(rec_labels, rec_feature_nodes, rec_L, rec_comp_T)
                        rec_sg.visualize(os.path.join(received_folder, str(seq_file_num) + ".png"))
                        received_scene_graphs[key][seq_file_num] = rec_sg
                        caption = self.text_gen.scene_graph_to_prompt(rec_sg)
                        dest_filename = os.path.join(ebno_folder, str(seq_file_num) + ".txt")
                        with open(dest_filename, "w") as f:
                            f.write(caption)
                        correctly_transmitted += 1
                except Exception:
                    print("\nTruncated file", file=sys.stderr)
                    continue
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
        return  scene_graph_dataset, correctly_transmitted, total_file_nb
    
def main(learning_filename, pipe, ebno):    
    sg_dataset, c_transmitted, total_file_nb = pipe.e2e(ebno)
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
        return labels, preds, c_transmitted, total_file_nb
        
            
if __name__ == "__main__":
    extraction_filename = "../Config/pipeline_extraction.yaml"
    learning_filename = "../Config/pipeline_learning.yaml"
    extraction_config = configuration(extraction_filename, from_function=True)
    correct_transmission, total_files = [], []
    df, sem_fid_df = pd.DataFrame(), pd.DataFrame()
    pipe = GBSED(extraction_config)
    for e in np.linspace(0, 20, 11):
        ebno = tf.constant(e, tf.float32)
        for i in range(1):
            main_return = main(learning_filename, pipe, ebno)
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
                        ), axis=0)
                df.to_csv("../../Data/Outputs/outputs.csv", index=False)
            sem_fid_df = pd.concat(
                (sem_fid_df, 
                 pd.DataFrame(
                     {
                         "iteration":[i], 
                         "ebno" : [e], 
                         "correct_transmission":[correctly_transmitted], 
                         "total_files":[total_file_nb]
                     }
                )), axis=0
            )
            sem_fid_df.to_csv("../../Data/Outputs/sem_fidelity.csv", index=False)