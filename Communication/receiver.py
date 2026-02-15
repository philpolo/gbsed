#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 19:40:56 2025

@author: polo

"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sionna.phy.utils import log10
from tensorflow.keras.layers import Layer, SeparableConv2D, Dense, LayerNormalization, ReLU

# carrier_frequency = 28e9
num_bits_per_symbol = 6
num_tx_ant = 2
num_rx_ant = 4

# # CGNN dimensions
num_nrx_iter = 8   # GNN iterations
d_s = 56           # Feature dimension
d_m = 56           # Message dimension

class NeuralReceiver(Layer):
    """CGNN-based MIMO receiver: updates both Rx and Tx node states."""
    def __init__(self):
        super().__init__()
        # 1) Initial embedding of the Rx nodes (Fig.3)
        self.init_conv = tf.keras.Sequential([
            SeparableConv2D(128, 3, padding='same'), LayerNormalization(epsilon=1e-6), ReLU(),
            SeparableConv2D(128, 3, padding='same'), LayerNormalization(epsilon=1e-6), ReLU(),
            SeparableConv2D(d_s,    3, padding='same'), LayerNormalization(epsilon=1e-6)
        ])
        # 2) Message MLP
        self.msg_mlp = tf.keras.Sequential([
            Dense(d_m, activation='relu'),
            Dense(d_m),
        ])
        # 3) State-update nets for Rx-nodes
        self.rx_update_nets = [
            tf.keras.Sequential([
                SeparableConv2D(256, 3, padding='same'), LayerNormalization(epsilon=1e-6), ReLU(),
                SeparableConv2D(256, 3, padding='same'), LayerNormalization(epsilon=1e-6), ReLU(),
                SeparableConv2D(d_s,   3, padding='same'), LayerNormalization(epsilon=1e-6)
            ]) for _ in range(num_nrx_iter)
        ]
        # 4) State-update nets for Tx-nodes
        self.tx_update_nets = [
            tf.keras.Sequential([
                SeparableConv2D(256, 3, padding='same'), LayerNormalization(epsilon=1e-6), ReLU(),
                SeparableConv2D(256, 3, padding='same'), LayerNormalization(epsilon=1e-6), ReLU(),
                SeparableConv2D(d_s,   3, padding='same'), LayerNormalization(epsilon=1e-6)
            ]) for _ in range(num_nrx_iter)
        ]
        # 5) Readout MLP on Tx states
        self.readout_mlp = tf.keras.Sequential([
            Dense(d_s, activation='relu'),
            Dense(num_bits_per_symbol),
        ])

    def call(self, y, h_hat, no):
        # y: [B, RX, S, C], h_hat: [B, RX, 1, TX, S, C]
        B = tf.shape(y)[0]
        RX = tf.shape(y)[1]
        S  = tf.shape(y)[2]
        C  = tf.shape(y)[3]
        # Embed noise & channel
        no_log = tf.tile(tf.reshape(log10(no), [B,1,1,1]), [1, RX, S, C])
        h     = tf.squeeze(h_hat, axis=2)  # [B, RX, TX, S, C]
        h_emb = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)
        # Initial Rx embedding
        x       = tf.stack([tf.math.real(y), tf.math.imag(y), no_log], -1)  
        x_flat  = tf.reshape(x, [B*RX, S, C, 3])                             
        rx_flat = self.init_conv(x_flat)                                    
        rx      = tf.reshape(rx_flat, [B, RX, S, C, d_s])                   
        # Initialize Tx state
        tx = tf.zeros([B, num_tx_ant, S, C, d_s], dtype=rx.dtype)
        bs = B * RX * num_tx_ant
        # Unrolled message-passing
        for i in range(num_nrx_iter):
            # tile & reshape for message computation
            tx_e = tf.reshape(
                tf.tile(tf.expand_dims(tx,1), [1, RX, 1, 1, 1, 1]),
                [bs, S, C, d_s]
            )
            rx_e = tf.reshape(
                tf.tile(tf.expand_dims(rx,2), [1, 1, num_tx_ant, 1, 1, 1]),
                [bs, S, C, d_s]
            )
            # Correct channel reshape without tile
            h_e  = tf.reshape(h_emb, [bs, S, C, 2])                         
            inp  = tf.reshape(tf.concat([tx_e, rx_e, h_e], -1), [-1, 2*d_s+2])
            m    = self.msg_mlp(inp)
            m    = tf.reshape(m, [B, RX, num_tx_ant, S, C, d_m])         
            # 1) Update Rx
            agg_rx = tf.reduce_sum(m, axis=2)                              
            st_in  = tf.reshape(tf.concat([rx, agg_rx], -1), [B*RX, S, C, d_s+d_m])
            upd_rx = self.rx_update_nets[i](st_in)
            upd_rx = tf.reshape(upd_rx, [B, RX, S, C, d_s])
            rx    += upd_rx
            # 2) Update Tx
            agg_tx = tf.reduce_sum(m, axis=1)                              
            st_in  = tf.reshape(tf.concat([tx, agg_tx], -1), [B*num_tx_ant, S, C, d_s+d_m])
            upd_tx = self.tx_update_nets[i](st_in)
            upd_tx = tf.reshape(upd_tx, [B, num_tx_ant, S, C, d_s])
            tx    += upd_tx
        # Final readout on Tx states
        tx_flat = tf.reshape(tx, [B * num_tx_ant, S, C, d_s])               
        out     = tf.reshape(tx_flat, [-1, d_s])                           
        llr     = self.readout_mlp(out)                                     
        return tf.reshape(llr, [B, num_tx_ant, S, C, num_bits_per_symbol])  
