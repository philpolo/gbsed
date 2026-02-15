#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 19:44:51 2025

@author: polo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sionna.phy import Block
from receiver import NeuralReceiver
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

from sionna.phy.utils import ebnodb2no
from sionna.phy.mapping import Mapper, BinarySource
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, ApplyOFDMChannel
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, RemoveNulledSubcarriers


class MIMOE2EModel(Block):
    def __init__(self, 
                 carrier_frequency=28e9, 
                 num_ofdm_symbols=14, 
                 fft_size=132, 
                 subcarrier_spacing=240e3, 
                 num_ut=1, 
                 num_tx_ant=2, 
                 num_rx_ant=4, 
                 num_bits_per_symbol=6, 
                 channel_model='C', 
                 coderate=0.5):
        super().__init__()
        # Resource grid
        self.rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=num_ut,
            num_streams_per_tx=num_tx_ant,
            cyclic_prefix_length=9,
            num_guard_carriers=[0,0],
            dc_null=False,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11]
        )
        self.coderate = coderate
        self.num_ut = num_ut
        self.num_tx_ant = num_tx_ant
        self.num_bits_per_symbol=num_bits_per_symbol
        self.data_syms = [i for i in range(num_ofdm_symbols) if i not in [2, 11]]
        self.n = int(self.rg.num_data_symbols * num_bits_per_symbol)
        self.k = int(self.n * self.coderate)
        # Antenna arrays & channel preinit
        self.ut_array = AntennaArray(1, num_tx_ant//2, 'dual', 'cross', '38.901', carrier_frequency) #1, "single", "V", "omni"
        self.bs_array = AntennaArray(1, num_rx_ant//2, 'dual', 'cross', '38.901', carrier_frequency)
        self._cdl = CDL(model=channel_model, delay_spread=100e-9,
                        carrier_frequency=carrier_frequency,
                        ut_array=self.ut_array, bs_array=self.bs_array,
                        direction='uplink', min_speed=0.0, max_speed=33.3)

        self._frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
        self._channel = ApplyOFDMChannel(add_awgn=True)


        # Coding & mapping
        self.src = BinarySource() # Keep for internal use if needed, but not for direct input
        self.enc = LDPC5GEncoder(self.k, self.n)
        self.dec = LDPC5GDecoder(self.enc, hard_out=True)
        self.map = Mapper('qam', num_bits_per_symbol)
        self.rg_map = ResourceGridMapper(self.rg)

        # Estimation
        self.ls = LSChannelEstimator(self.rg, interpolation_type='lin')
        self.rm = RemoveNulledSubcarriers(self.rg)
        # Neural receiver
        self.neural_rx = NeuralReceiver()


    @tf.function
    def call(self, batch_size, ebno_db, input_bits=None):
        # self.new_topology(batch_size)
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, self.rg)

        ####################################
        #Transmitter
        # Calculate the total number of information bits for all streams in the batch
        total_info_bits_per_batch = batch_size * self.num_ut * self.num_tx_ant * self.k

        # Use provided input_bits if available, otherwise generate random bits
        if input_bits is not None:
            # Ensure input_bits is a TensorFlow tensor and cast to float32
            input_bits = tf.cast(input_bits, tf.float32)
            # Check if the length of input_bits matches the expected total bits
            tf.Assert(tf.equal(tf.shape(input_bits)[0], total_info_bits_per_batch),
                      ["Input bits length does not match expected total bits.",
                       "Expected:", total_info_bits_per_batch, "Got:", tf.shape(input_bits)[0]])
            # Reshape input_bits to match the encoder's expected input shape [effective_batch_size, k]
            b_enc_in = tf.reshape(input_bits, [batch_size * self.num_ut * self.num_tx_ant, self.k])
            # Reshape original bits for return value `b`
            b = tf.reshape(input_bits, [batch_size, self.num_ut, self.num_tx_ant, self.k])
        else:
            # Fallback to BinarySource if no input_bits are provided
            b = self.src([batch_size, self.num_ut, self.num_tx_ant, self.k])
            b_enc_in = tf.reshape(b, [batch_size * self.num_ut * self.num_tx_ant, self.k])


        c_enc_out = self.enc(b_enc_in) # Coded bits: [batch_size * num_ut * num_tx_ant, self.n]

        # Reshape coded bits for the Mapper
        num_symbols_per_stream = self.n // self.num_bits_per_symbol
        c_map_in = tf.reshape(c_enc_out, [-1, self.num_bits_per_symbol]) # Shape: [total_symbols, num_bits_per_symbol]

        x_map_out = self.map(c_map_in) # Modulated symbols: [total_symbols, 1] (complex)

        # Reshape modulated symbols for the ResourceGridMapper
        x_rg_map_in = tf.reshape(x_map_out, [batch_size, self.num_ut, self.num_tx_ant, num_symbols_per_stream])
        x_rg = self.rg_map(x_rg_map_in)

        #####################################
        # Channel
        a,tau = self._cdl(batch_size, self.rg.num_ofdm_symbols, 1/self.rg.ofdm_symbol_duration)
        hfreq = cir_to_ofdm_channel(self._frequencies, a, tau, normalize=True)
        y       = self._channel(tf.expand_dims(x_rg,1), hfreq, no)

        #####################################
        #Receiver
        h_hat,_ = self.ls(y, no)
        y_cl    = tf.squeeze(y,1)
        h_cl = tf.squeeze(h_hat,1)
        llr     = self.neural_rx(y_cl, h_cl, no)                    # [B, TX, S, C, bits]
        llr   = tf.gather(llr, self.data_syms, axis=2)
        # Reshape LLRs for the decoder
        # The total number of LLRs must match total_info_bits_per_batch after decoding
        llr   = tf.reshape(llr, [batch_size, self.num_ut, self.num_tx_ant, self.n])
        b_hat = self.dec(llr) #llr_f
        return b, b_hat

if __name__ == "__main__":
    model = MIMOE2EModel()
    ebno = tf.constant(20.0, tf.float32)
    b, b_hat = model(1, ebno)