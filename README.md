# Multi-Marginal Stochastic Flow Matching for High-Dimensional Snapshot Data at Irregular Time Points

[](https://icml.cc/Conferences/2025)
[](https://arxiv.org) [](https://opensource.org/licenses/MIT)

[cite\_start]Official PyTorch implementation for the ICML 2025 paper "Multi-Marginal Stochastic Flow Matching for High-Dimensional Snapshot Data at Irregular Time Points" by Justin Lee, Behnaz Moradijamei, and Heman Shakeri[cite: 3, 17].



MMSFM learns the continuous-time dynamics of a system from a few snapshots in time, even if they are unevenly spaced. It works by:

1.  [cite\_start]**Aligning Snapshots:** We use a first-order Markov approximation of a Multi-Marginal Optimal Transport (MMOT) plan to find correspondences between data points in consecutive snapshots[cite: 132].
2.  [cite\_start]**Creating Continuous Paths:** We use these aligned points as control points for transport splines[cite: 110]. [cite\_start]Specifically, we use monotonic cubic Hermite splines to create smooth, well-behaved paths ($\\mu\_t$) that interpolate between the snapshots[cite: 197]. [cite\_start]This method avoids the "overshooting" artifacts that can occur with natural cubic splines, especially with irregular time intervals[cite: 225].
3.  [cite\_start]**Learning Dynamics with Overlapping Flows:** Instead of learning a single, global flow, we train a single neural network on "mini-flows" defined over small, overlapping windows of snapshots (e.g., triplets like $\\rho\_i, \\rho\_{i+1}, \\rho\_{i+2}$)[cite: 148]. [cite\_start]This approach improves the model's robustness and prevents overfitting to sparse data[cite: 83].
4.  **Simulation-Free Training:** The entire process is simulation-free. [cite\_start]We train our drift and score networks by directly regressing them against the analytical targets derived from our spline-based probability paths, making the training process highly efficient[cite: 82, 186].

The result is a single, continuous model of the system's dynamics that can generate new trajectories and sample states at any arbitrary time point $t \\in [0, 1]$.

> *Example trajectories for a 32x32 pixel image progression through the Imagenette classes (gas pump $\\to$ golf ball $\\to$ parachute). [cite\_start]Results are generated using our Triplet ($k=2$) model with an equidistant time scheme[cite: 262, 268].*

## Repository Structure

```
.
├── data/                 # Scripts to download and preprocess datasets
├── notebooks/            # Jupyter notebooks for visualization and analysis
├── src/
│   ├── models.py         # Core MMSFM model and network architectures
│   ├── dataloaders.py    # Data loading utilities
│   └── splines.py        # Implementation of transport splines
├── train.py              # Main script to train a new model
├── evaluate.py           # Script to evaluate a trained model and generate trajectories
└── environment.yml       # Conda environment file
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Shakeri-Lab/MMSFM.git
    cd MMSFM
    ```

2.  **Create Conda Environment:**
    We recommend using Conda to manage dependencies. We used Python 3.10 to develop our code.

    ```bash
    ## Create in default venv directory
    conda create -n mmsfmvenv python=3.10

    ## OR create in current directory
    conda create -p ./mmsfmvenv python=3.10
    ```
    Then activate using either `conda activate mmsfmvenv` or `conda activate ./mmsfmvenv` depending on the virtual environment location.

3. **Installation:**
    Run `make_venv.sh` which will install the necessary packages. It will first download `MIOFlow` and `torchcfm` from their respective GitHub repositories. In particular, the script will download the specific archived commits from the respective `MIOFlow` and `torchcfm` packages that we used at the time of development in order to maintain reproducability. We also pin the specific versions of each package in `requirements.txt` for the same reason. Next, the script will install the packages in `requirements.txt`, followed by the `MIOFlow`, `torchcfm`, and our code. These latter three packages will be installed in editable mode.

    The `MIOFlow` commit hash is `1b09f2c7eefefcd75891d44bf86c00a4904a0b05`.

    The `torchcfm` commit hash is `af8fec6f6dc3a0dc7f8fb25d2ee0ca819fa5412f`.

    [cite\_start]Our implementation uses PyTorch, POT (Python Optimal Transport), and torchsde[cite: 596, 599, 613].

4.  **Download Data:**
    Run the provided scripts to download and preprocess the datasets used in the paper.

    ```bash
    python -m data.download_all
    ```

## Running Experiments

You can train a new MMSFM model using `train.py`.

**Example: Training the Triplet model on Imagenette**

```bash
python train.py \
    --model Triplet \
    --dataset Imagenette \
    --data_path ./data/imagenette \
    --batch_size 64 \
    --lr 1e-4 \
    --epochs 250
```

**Example: Generating trajectories from a trained model**

```bash
python evaluate.py \
    --checkpoint_path ./checkpoints/imagenette_triplet.pt \
    --num_samples 16 \
    --output_dir ./results/imagenette_trajs
```

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{lee2025mmsfm,
  title={Multi-Marginal Stochastic Flow Matching for High-Dimensional Snapshot Data at Irregular Time Points},
  author={Lee, Justin and Moradijamei, Behnaz and Shakeri, Heman},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025},
  series={Proceedings of Machine Learning Research},
  volume={267},
  publisher={PMLR}
}
```
