# GBSED — Graph-Based Semantic Encoder–Decoder for Task-Oriented Communications in Connected Autonomous Vehicles

## Overview

GBSED is a semantic communication framework designed for **task-oriented transmission of driving scene information** between Connected Autonomous Vehicles (CAVs). Instead of transmitting raw sensor data or camera images, GBSED represents road environments as compact **scene graphs** that capture the semantic and spatial relationships between road entities.

These graph-based representations are **semantically compressed and transmitted over a simulated wireless channel**, significantly reducing communication overhead compared to conventional data transmission approaches. The communication pipeline is integrated with a **MIMO–OFDM physical layer communication system operating under realistic 3GPP CDL channel conditions**, enabling the evaluation of semantic transmission performance in practical wireless environments.

At the receiver side, the transmitted scene graphs are **decoded and directly exploited for downstream driving tasks**, such as risk assessment and decision support, without requiring full reconstruction of the original raw data.


---

## Architecture

```
Raw Driving Images
        │
        ▼
  Scene Graph Extraction  (roadscene2vec)
        │
        ▼
  sg_autoencoder.encode()    ──►  Adjacency tensor T  (nodes × nodes × relations)
        │
        ▼
  sem_compression()          ──►  Removes zero-slice relation matrices → compact (comp_T, L)
        │
        ▼
  Binary Serialization       ──►  float16 → bit array
        │
        ▼
  MIMO-OFDM E2E Channel      ──►  LDPC encoding, QAM mapping, CDL-C fading channel
  (MIMOE2EModel / Sionna)          Neural receiver decodes received signal
        │
        ▼
  Binary Deserialization     ──►  Recovered float16 array
        │
        ▼
  sem_decompression()        ──►  Reconstruct full adjacency tensor
        │
        ▼
  sg_autoencoder.decode()    ──►  Reconstruct SceneGraph object
        │
        ▼
  Downstream Task Inference  ──►  MRGCN / MRGIN graph classifier (roadscene2vec)
        │
        ▼
  Outputs (labels, predictions, semantic fidelity metrics)
```

---

## Project Structure

```
├── Data/
|       ├── 1043/                # 1043-syn dataset (see link below)
├── gbsed-main/
        ├── pipeline/
        │   └── pipeline.py              # Main GBSED class and end-to-end pipeline
        ├── sgautoencoder/
        │   └── sg_autoencoder.py        # Scene graph encoder/decoder and semantic compression
        ├── Communication/
        │   ├── e2emodel.py              # MIMO-OFDM end-to-end model (Sionna)
        │   ├── receiver.py              # Neural receiver for OFDM channel estimation
        │   └── weights/
        │       └── neural_rx_ofdm_mimo_cdl_final.h5
        ├── learning/
        │   ├── rs2vec_training.py       # Training utilities for roadscene2vec
        │   ├── graph_profile_metrics.txt
        │   └── rs2vec_training_loss.csv
        ├── utils/
        │   └── datasetGenerator.py      # sg2text caption generator
        ├── Config/
        │   ├── pipeline_extraction.yaml # Config for scene graph extraction
        │   └── pipeline_learning.yaml   # Config for downstream learning
        └── requirements.txt
```
Note: The 1043-syn dataset can be found at: <https://ieee-dataport.org/documents/scenegraph-risk-assessment-dataset>

---

## Installation

All experiments were performed on **Ubuntu 24.04** with **Python 3.12**.

**1. Create and activate a Conda environment**

```bash
conda create --name av python=3.12
conda activate av
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch` (2.8.0+cu126) and `torch-geometric` — graph neural network training
- `tensorflow` (2.19.0) and `sionna` (1.1.0) — physical layer simulation
- `roadscene2vec` — scene graph extraction and graph-based learning
- `detectron2` — object detection for scene graph extraction from real images
- `transformers`, `diffusers` — language model utilities for scene captioning
- `wandb` — experiment tracking
- [roadscene2vec](https://github.com/AICPS/roadscene2vec) library is used for scene graph extraction, dataset management, and graph-based learning. The library must be installed and available in the Python path. The `roadscene2vec` module is imported via `sys.path` manipulation in the source files; ensure it is located at the expected relative path (`../../roadscene2vec` or installed as a package).

> **Note:** The full dependency list is large and GPU-specific (CUDA 12.6). It is strongly recommended to use a machine with a compatible NVIDIA GPU.

---

## Configuration

Two YAML config files control the pipeline:

### `Config/pipeline_extraction.yaml`

Controls scene graph extraction and the communication pipeline. Key fields:

| Field | Description |
|---|---|
| `loading_type` | `pickle` (load pre-extracted dataset) or `folder` (extract from raw images) |
| `dataset_type` | `image` or `carla` |
| `location_data.input_path` | Path to the input dataset (`.pkl` or folder) |
| `location_data.data_save_path` | Path to save the reconstructed transfer dataset |
| `relation_extraction_settings` | Actor types, relation types, thresholds, and proximity rules |

### `Config/pipeline_learning.yaml`

Controls the downstream graph classification task. Key fields:

| Field | Description |
|---|---|
| `model_configuration.model` | `mrgcn` or `mrgin` |
| `model_configuration.model_load_path` | Path to a pretrained model checkpoint |
| `training_configuration.task_type` | `sequence_classification` or `collision_prediction` |
| `training_configuration.epochs` | Number of training epochs |

---

## Usage

### Train the risk assessment model

From the `learning/` directory:

```bash
python learning.py
```

### Running the Full End-to-End Pipeline

From the `pipeline/` directory:

```bash
python pipeline.py
```

This will:
1. Load the scene graph dataset from the path specified in `pipeline_extraction.yaml`
2. For each scene graph, encode it into a binary representation
3. Transmit the bits over the simulated MIMO-OFDM CDL channel across a range of Eb/N₀ values (0–20 dB)
4. Decode the received bits and reconstruct scene graphs
5. Run inference using the pretrained graph classifier
6. Save prediction results to `../../Data/Outputs/outputs.csv` and semantic fidelity metrics to `../../Data/Outputs/sem_fidelity.csv`
7. For each reconstructed scene graph, save its textual description in the correct folder

---

## Communication Model Details

The `MIMOE2EModel` (in `Communication/e2emodel.py`) simulates a realistic 5G mmWave uplink with the following default configuration:

| Parameter | Value |
|---|---|
| Carrier frequency | 28 GHz |
| Channel model | CDL-C (3GPP TR 38.901) |
| OFDM symbols | 14 |
| FFT size | 132 subcarriers |
| Subcarrier spacing | 240 kHz |
| TX antennas | 2 |
| RX antennas | 4 |
| Modulation | 64-QAM (6 bits/symbol) |
| Channel coding | LDPC (5G NR), rate 0.5 |
| Channel estimation | Neural receiver (pretrained) |

---

## Outputs

After a full pipeline run, the following outputs are produced:

- **`outputs.csv`** — per-frame classification results with columns: `iteration`, `label`, `prediction`, `ebno`
- **`sem_fidelity.csv`** — semantic fidelity metrics per Eb/N₀ value: `correctly_transmitted` / `total_files`
- **Reconstructed scene graph visualizations** — saved as `.png` files in the dataset directory
- **Scene captions** — `.txt` files containing natural language descriptions of received scene graphs


---

## Citation

If you use this work, please cite the corresponding publication:

[Graph Based Semantic Encoder Decoder Framework for Task Oriented Communications in Connected Autonomous Vehicles](https://arxiv.org/pdf/2603.08438)

```bibtex
@article{ribouh2026graph,
  title={Graph Based Semantic Encoder Decoder Framework for Task Oriented Communications in Connected Autonomous Vehicles},
  author={Ribouh, Soheyb and Di Ngoma, Phil Polo Ditsia},
  journal={arXiv preprint arXiv:2603.08438},
  year={2026}
}



```markdown
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
