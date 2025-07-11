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
├── data/
│   └── datagen.py                    # Script to download and preprocess datasets
├── mmsfm/
│   ├── models/
│   │   └── models.py                 # Network architectures
│   ├── multimarginal_cfm.py          # Core implementation of multi-marginal flow matcher w/ splines
│   └── multimarginal_otsampler.py    # Implementation of (ordered) multi-marginal optimal transport
├── scripts/                          # Contains scripts for training models
│   ├── main.py                       # Core training script for synthetic and single-cell data
│   ├── modelagent.py                 # Contains logic for training, evaluation, and inference
│   ├── plotter.py                    # Plotting utilities for training losses, evaluations, and trajectories
│   ├── utils.py
│   └── images/                       # Contains scripts for training models on image datasets
│       ├── images_main.py            # Core training script for class progression task on image data
│       ├── images_train.py           # Contains training logic
│       ├── images_eval.py            # Contains inference logic
│       ├── images_plot.py            # Contains plotting logic for losses and trajectories
│       └── images_utils.py
├── README.md
├── pyproject.toml
├── requirements.txt                  # Environment file w/ all package versions pinned
├── make_venv.sh                      # Helper script to install this package
├── runner.sh                         # Helper script to call scripts/main.py
├── image_runner.sh                   # Helper script to call scripts/images/image_main.py
└── .gitignore
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
    conda activate mmsfmvenv

    ## OR create in current directory
    conda create -p ./mmsfmvenv python=3.10
    conda activate ./mmsfmvenv
    ```

3. **Installation:**
    Run [`make_venv.sh`](make_venv.sh) which will install the necessary packages.
    It will first download `MIOFlow` and `torchcfm` from their respective GitHub repositories.
    In particular, the script will download the specific archived commits from the respective `MIOFlow` and `torchcfm` packages that we used at the time of development in order to maintain reproducability.
    We also pin the specific versions of each package in `requirements.txt` for the same reason.
    Next, the script will install the packages in `requirements.txt`, followed by the `MIOFlow`, `torchcfm`, and our code.
    These latter three packages will be installed in editable mode.

    The `MIOFlow` commit hash is `1b09f2c7eefefcd75891d44bf86c00a4904a0b05`.

    The `torchcfm` commit hash is `af8fec6f6dc3a0dc7f8fb25d2ee0ca819fa5412f`.

    [cite\_start]Our implementation uses PyTorch, POT (Python Optimal Transport), and torchsde[cite: 596, 599, 613].

4.  **Download Data:**
    In order to use the single cell datasets for CITEseq and Multiome, you will first need to download the following files from the
    [Multimodal Single-Cell Integration](https://kaggle.com/competitions/open-problems-multimodal) Kaggle competition:
    - `metadata.csv`
    - `train_cite_inputs.h5`
    - `test_cite_inputs.h5`
    - `train_multi_targets.h5`

    These files must be saved to `data/`.

    The [`data/datagen.py`](data/datagen.py) script can do 3 things:

    1. Generate the synthetic datasets and draw the corresponding scatter plots.
    2. Preprocess the single cell datasets using the top 50 and 100 PCA components, as well as the top 1000 highly variable genes.
    3. Download if necessary, then preprocess the CIFAR-10 and Imagenette datasets for easier loading. Downloading is handled via Torchvision's in-built datasets.

    You can run the script as follows:

    ```bash
    cd data
    ## You should be located at <rootdir>/MMSFM/data/

    ## To only generate synthetic data
    python datagen.py --datasets synth

    ## To only preprocess single cell data
    python datagen.py --datasets real

    ## To only download and preprocess CIFAR-10 and Imagenette data
    python datagen.py --datasets images

    ## To do all
    python datagen.py --datasets synth real images
    ```
    You should only have to run this script once, given an issue with how Torchvision checks whether a dataset has already been downloaded for Imagenette.
    [See here](https://github.com/pytorch/vision/pull/8638) for the relevant issue.

## Running Experiments

### Synthetic and Single-cell Data
You can train a new MMSFM model for the synthetic and single-cell data using [`scripts/main.py`](scripts/main.py).
Given the large number of possible arguments, we provide a simple runner script in [`runner.sh`](runner.sh) where you can easily set the desired hyperparameters.
Don't forget to update the `WANDBARGS` in `runner.sh` to either include your entity and project names, or to set the `--no_wandb` flag to disable wandb for that run.
Whether you choose to directly call `scripts/main.py` or use `runner.sh`, please do so from the base directory `<rootdir>/MMSFM/`.
Either way, you will train the model, generate some sample trajectories, and create some evaluation and visualization plots.

**Example: Training the Triplet model on S-shaped Gaussians**
```bash
## pwd shoud output <rootdir>/MMSFM/

## Directly calling scripts/main.py
python scripts/main.py \
    --dataname sg \
    --flowmatcher sb \
    --agent_type triplet \
    --spline cubic \
    --modelname mlp \
    --batch_size 64 \
    --n_steps 1000 \
    --n_epochs 5 \
    --lr 1e-4 \
    --zt 0 1 2 3 4 5 6 \
    --no_wandb \
    --outdir sg

## Using the provided helper runner script
./runner.sh
```

### CIFAR-10 and Imagenette Data
Given some differences in the datatypes (especially size of the data) as well as evaluations and plots, we provide a second script for training a MMSFM model for the image datasets found at [`scripts/images/images_main.py`](scripts/images/images_main.py).
Likewise, we also provide a simple runner script in [`image_runner.sh`](image_runner.sh), which also contains a `WANDBARGS` argument list as well as a `no_wandb` flag.
Again, please call either the python script or runner script from the base directory `<rootdir>/MMSFM/`.

This version of the trainer additionally implements accumulated gradients as well as a method to checkpoint and resume training.
We submitted jobs using the Slurm job scheduler, which gave us access to the remaining walltime.
We used this information to programatically exit the training loop and set up a checkpoint to prevent timeout issues.
If not submitting jobs through Slurm, we assume the remaining walltime is practically unlimited at 999 days.

**Example: Training the Triplet model on CIFAR-10**
```bash
## pwd shoud output <rootdir>/MMSFM/

## Directly calling scripts/images/images_main.py
python scripts/images/images_main.py \
    train eval plot \
    --dataname cifar10 \
    --size 32 \
    --window_size 2 \
    --spline cubic \
    --monotonic \
    --score_matching \
    --zt 0 1 2 3 \
    --progression 2 4 6 8 \
    --batch_size 16 \
    --accum_steps 2 \
    --n_steps 20 \
    --n_epochs 10 \
    --lr 1e-8 1e-4 \
    --save_interval 2 \
    --ckpt_interval 2 \
    --no_wandb \
    --outdir cifar10

## Using the provided helper runner script
./image_runner.sh
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
