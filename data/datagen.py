from os.path import exists as os_exists
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scanpy as sc
import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
import joblib


####################### Synth Data Gen Functions #######################
########################################################################
def three_gaussians(N, mus, sigmas, seed=1000):
    prng = np.random.default_rng(seed=seed)
    Xs = np.zeros((3, N, 2))
    for i in range(3):
        mu_t = mus[i]
        sigma_t = sigmas[i]
        cov_t = (sigma_t ** 2) * np.eye(2)
        X_t = prng.multivariate_normal(mu_t, cov_t, size=(N,))
        Xs[i] = X_t
    return Xs


def gaussian_checker_moons(N, noise=None, seed=1000):
    prng = np.random.default_rng(seed=seed)
    Xs = np.zeros((3, N, 2))

    ## make standard MVN
    mu = np.zeros(2)
    cov = 0.25 * np.eye(2)
    X_0 = prng.multivariate_normal(mu, cov, size=(N,))

    ## make checkerboard in the following order:
    ## + - - - - +
    ## |   6   8 |
    ## | 2   4   |
    ## |   5   7 |
    ## | 1   3   |
    ## + - - - - +
    X_i = np.zeros((N, 2))
    N_block = N // 8
    base_xrange = np.array([-1., -0.5])
    base_yrange = np.array([-1., -0.5])
    k = 0
    for offset_xy in [0, 0.5]:
        for offset_x in [0., 1.]:
            for offset_y in [0., 1.]:
                xrange = base_xrange + offset_xy + offset_x
                yrange = base_yrange + offset_xy + offset_y
                x = prng.uniform(low=xrange[0], high=xrange[1], size=(N_block,))
                y = prng.uniform(low=yrange[0], high=yrange[1], size=(N_block,))
                start = k * N_block
                end = start + N_block
                X_i[start:end, 0] = x
                X_i[start:end, 1] = y
                k += 1

    ## make moons
    X_1, _ = make_moons(n_samples=N, shuffle=True, noise=noise, random_state=seed)

    for i, X_t in enumerate([X_0, X_i, X_1]):
        _idx = np.arange(N)
        prng.shuffle(_idx)
        Xs[i, :, 0] = X_t[_idx, 0]  # type: ignore
        Xs[i, :, 1] = X_t[_idx, 1]  # type: ignore
    return Xs


def s_gaussians(N, seed=1000):
    prng = np.random.default_rng(seed=seed)
    Xs = np.zeros((7, N, 2))

    mus = np.zeros((7, 2))
    mus[0] = [0, 0]
    mus[1] = [1, 4]
    mus[2] = [5, 4]
    mus[3] = [6, 0]
    mus[4] = [7, -4]
    mus[5] = [11, -4]
    mus[6] = [12, 0]

    cov = np.eye(2) * 0.5
    for i in range(7):
        mu = mus[i]
        X_t = prng.multivariate_normal(mu, cov, size=(N,))
        Xs[i] = X_t

    return Xs


def alpha_gaussians(N, seed=1000):
    prng= np.random.default_rng(seed=seed)
    Xs = np.zeros((7, N, 2))

    mus = np.zeros((7, 2))
    mus[0] = [6, 6]
    mus[1] = [2, 6]
    mus[2] = [-3, 0]
    mus[3] = [-6, 3]
    mus[4] = [-3, 6]
    mus[5] = [2, 0]
    mus[6] = [6, 0]

    cov = np.eye(2) * 0.5
    for i in range(7):
        mu = mus[i]
        X_t = prng.multivariate_normal(mu, cov, size=(N,))
        Xs[i] = X_t

    return Xs
########################################################################
########################################################################


################### CITEseq & Multiome Preprocessing ###################
########################################################################
def load_citeseq():
    ## load both train and test because
    ## train only has times [2, 3, 4]
    ## and test has time [7]
    metadata = pd.read_csv('metadata.csv')
    cite_df = pd.read_hdf('train_cite_inputs.h5')
    cite_df2 = pd.read_hdf('test_cite_inputs.h5')

    idx = metadata['technology'].str.contains('citeseq')
    cite_meta = metadata[idx]
    cite_meta = cite_meta[cite_meta['donor'] == 13176]

    cite_df_cell_ids = set(cite_df.index)  # type: ignore
    cite_df2_cell_ids = set(cite_df2.index)  # type: ignore
    ts = [2, 3, 4, 7]

    nt = {}  # get counts at each time
    cite_df_all = pd.DataFrame(
        index=pd.Index(data=[], name='cell_id'),
        columns=cite_df.columns  # type: ignore
    )
    for t in ts:
        t_idx = cite_meta['day'] == t
        nt[t] = t_idx.sum()
        cite_meta_t = cite_meta[t_idx]
        cell_ids = set(cite_meta_t['cell_id'])
        valid_ids = list(cite_df_cell_ids & cell_ids)
        valid_ids2 = list(cite_df2_cell_ids & cell_ids)
        cite_df_t = cite_df.loc[valid_ids]  # type: ignore
        cite_df2_t = cite_df2.loc[valid_ids2]  # type: ignore
        cite_df_all = pd.concat([cite_df_all, cite_df_t, cite_df2_t])

    nt_vals = [nt[t] for t in sorted(nt.keys())]
    nt_vals_cumsum = np.cumsum(nt_vals)

    return cite_df_all, nt_vals_cumsum


def load_multiome():
    metadata = pd.read_csv('metadata.csv')
    mult_df = pd.read_hdf('train_multi_targets.h5')

    idx = metadata['technology'].str.contains('multiome')
    mult_meta = metadata[idx]
    mult_meta = mult_meta[mult_meta['donor'] == 13176]

    mult_df_cell_ids = set(mult_df.index)  # type: ignore
    ts = [2, 3, 4, 7]

    nt = {}  # get counts at each time
    mult_df_all = pd.DataFrame(
        index=pd.Index(data=[], name='cell_id'),
        columns=mult_df.columns  # type: ignore
    )
    for t in ts:
        t_idx = mult_meta['day'] == t
        nt[t] = t_idx.sum()
        mult_meta_t = mult_meta[t_idx]
        cell_ids = set(mult_meta_t['cell_id'])
        valid_ids = list(mult_df_cell_ids & cell_ids)
        mult_df_t = mult_df.loc[valid_ids]  # type: ignore
        mult_df_all = pd.concat([mult_df_all, mult_df_t])

    nt_vals = [nt[t] for t in sorted(nt.keys())]
    nt_vals_cumsum = np.cumsum(nt_vals)

    return mult_df_all, nt_vals_cumsum


def preprocess_pca(df_all, nt_vals_cumsum, n_components, seed=None):
    data_np = df_all.to_numpy()

    PCA_op = PCA(n_components=n_components)
    data_pca = PCA_op.fit_transform(data_np)
    data_pca_list = np.vsplit(data_pca, nt_vals_cumsum)[:-1]

    data_pca_train = []
    data_pca_test = []

    for data_pca_t in data_pca_list:
        _train, _test = train_test_split(
            data_pca_t, test_size=0.2, shuffle=True, random_state=seed
        )
        data_pca_train.append(_train)
        data_pca_test.append(_test)

    return data_pca_train, data_pca_test


def preprocess_hivars(df_all, nt_vals_cumsum, n_top_genes, seed=None):
    adata = sc.AnnData(df_all)

    tmp = sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False)
    hi_vars = tmp[tmp['highly_variable']].index  # type: ignore
    df_hi_vars = df_all[hi_vars]
    hivars_list = np.vsplit(df_hi_vars, nt_vals_cumsum)[:-1]

    hivars_train = []
    hivars_test = []

    for hivars_t in hivars_list:
        _train, _test = train_test_split(
            hivars_t, test_size=0.2, shuffle=True, random_state=seed
        )
        hivars_train.append(_train)
        hivars_test.append(_test)

    return hivars_train, hivars_test


def save_sc_data(train_list, test_list, name):
    splitnames = ['train', 'test']
    for _list, splitname in zip([train_list, test_list], splitnames):
        np.savez(
            f'{name}_{splitname}.npz',
            t0=_list[0],
            t1=_list[1],
            t2=_list[2],
            t3=_list[3]
        )
########################################################################
########################################################################


################# CIFAR10 and Imagenette Preprocessing #################
########################################################################
def load_cifar10(root='./cifar10/', download=False):
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=download
    )

    ## No need to download if trainset was downloaded
    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        transform=transform,
        download=False
    )

    classes = (
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )

    return trainset, testset, classes


def load_imagenette(size, root='./imagenette/', download=False):
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((size, size)),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.Imagenette(
        root=root,
        split='train',
        size='full',
        download=download,
        transform=transform
    )

    ## No need to download if trainset was downloaded
    testset = torchvision.datasets.Imagenette(
        root=root,
        split='val',
        size='full',
        download=False,
        transform=transform
    )

    classes = (
        'tench', 'English springer', 'cassette player',
        'chain saw', 'church', 'French horn', 'garbage truck',
        'gas pump', 'golf ball', 'parachute'
    )

    return trainset, testset, classes


def group_by_class(dataset, classes, datasetname):
    tmp = [[] for _ in classes]
    for i in tqdm(range(len(dataset)), f'Group {datasetname} by class'):
        x, y = dataset[i]
        Xc = tmp[y]
        Xc.append(x)

    X = [torch.stack(Xc) for Xc in tmp]
    return X


def group_cifar10_by_class(download=False):
    trainset, testset, classes = load_cifar10(root='./cifar10/', download=download)
    train_by_class = group_by_class(trainset, classes, 'CIFAR10 Trainset')
    test_by_class = group_by_class(testset, classes, 'CIFAR10 Testset')

    joblib.dump(train_by_class, f'./cifar10/trainset.pkl')
    joblib.dump(test_by_class, f'./cifar10/testset.pkl')


def group_imagenette_by_class(size, download=False):
    trainset, testset, classes = load_imagenette(size, root='./imagenette/', download=download)
    train_by_class = group_by_class(trainset, classes, f'Imagenette{size} Trainset')
    test_by_class = group_by_class(testset, classes, f'Imagenette{size} Testset')

    joblib.dump(train_by_class, f'./imagenette/trainset{size}.pkl')
    joblib.dump(test_by_class, f'./imagenette/testset{size}.pkl')
########################################################################
########################################################################


######################## Preprocessing Runners #########################
########################################################################
def create_plot_synth_datasets():
    N = 20000     # N samples
    noise = 0.05  # noise kwarg for sklearn make_moons()
    seed = 1000

    mus = np.zeros((3, 2))
    mus[0] = [0, 0]
    mus[1] = [3, 3]
    mus[2] = [6, 0]
    sigmas = np.ones(3)

    print('Creating 3 Gaussians...')
    threeg = three_gaussians(20000, mus, sigmas, seed=1000)
    print('Creating Gaussians to checker to moons...')
    gcm = gaussian_checker_moons(20000, noise=noise, seed=1000)
    print('Creating S-shaped Gaussians...')
    sg = s_gaussians(20000, seed=1000)
    sgtest = s_gaussians(2000, seed=2000)
    print('Creating alpha-shaped Gaussians...')
    alphag = alpha_gaussians(20000, seed=3000)
    alphagtest = alpha_gaussians(2000, seed=3001)

    ## Save datasets
    dnames = ['3g', 'gcm', 'sg', 'sgtest', 'alphag', 'alphagtest']
    D = [threeg, gcm, sg, sgtest, alphag, alphagtest]

    for d, dname in zip(D, dnames):
        np.save(f'{dname}.npy', d)

    ## Plot scatterplots
    colors = sorted(
        mcolors.TABLEAU_COLORS,
        key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c)))
    )

    titles = ['3 Gaussians', 'GCM',
              'S-shaped Gaussians Train', 'S-shaped Gaussians Test',
              r'$\alpha$-shaped Gaussians Train', r'$\alpha$-shaped Gaussians Test']
    fig = plt.figure(figsize=(8, 8))

    for d, dname, title in zip(D, dnames, titles):
        fig.clf()
        ax = fig.gca()
        fig.suptitle(title, fontsize=18)
        xmin, ymin = d.min(axis=(0, 1))
        xmax, ymax = d.max(axis=(0, 1))
        xmargin = (xmax - xmin) * 0.05
        ymargin = (ymax - ymin) * 0.05
        ax.set_xlim((xmin - xmargin, xmax + xmargin))
        ax.set_ylim((ymin - ymargin, ymax + ymargin))
        ax.set_xticks([])
        ax.set_yticks([])

        print(f'Plotting {dname} scatter plot...')
        for t in range(d.shape[0]):
            ax.scatter(
                d[t, :1000, 0], d[t, :1000, 1],
                color=colors[t],
                label=fr'$\rho_{t}$'
            )
        ax.legend()
        fig.tight_layout()
        fig.savefig(f'{dname}_scatter.png')


def preprocess_singlecell():
    ## Check that the prereq datafiles from Kaggle exist
    assert os_exists('metadata.csv'), 'metadata.csv is missing'
    assert os_exists('train_cite_inputs.h5'), 'train_cite_inputs.h5 is missing'
    assert os_exists('test_cite_inputs.h5'), 'test_cite_inputs.h5 is missing'
    assert os_exists('train_multi_targets.h5'), 'train_multi_targets.h5 is missing'

    ## Load and preprocess citeseq and multiome datasets
    savenames = []
    train_list_list = []
    test_list_list = []
    dnames = ['cite', 'mult']
    incr = 0  # for setting the train_test_split() random state

    for dname, load_fn in zip(dnames, [load_citeseq, load_multiome]):
        print(f'Loading {dname}...')
        df_all, nt_vals_cumsum = load_fn()
        for n_components in [50, 100]:
            print(f'  Computing pca {n_components}...')
            savenames.append(f'{dname}_pca{n_components}')
            pca_train, pca_test = preprocess_pca(
                df_all, nt_vals_cumsum, n_components, seed=1000+incr
            )
            train_list_list.append(pca_train)
            test_list_list.append(pca_test)
            incr += 1

        print('  Computing hivars...')
        savenames.append(f'{dname}_hivars')
        hivars_train, hivars_test = preprocess_hivars(
            df_all, nt_vals_cumsum, 1000, seed=1000+incr
        )
        train_list_list.append(hivars_train)
        test_list_list.append(hivars_test)
        incr += 1

    ## Save preprocessed datasets
    for train_list, test_list, savename in \
            zip(train_list_list, test_list_list, savenames):
        save_sc_data(train_list, test_list, savename)


def preprocess_images():
    ## call both load fns with download=True first
    ## because for some reason the download=True flag does not
    ## correctly load an already downloaded dataset for Imagenette
    ## github.com/pytorch/vision/pull/8638
    load_cifar10(download=True)
    load_imagenette(32, download=True)

    ## Group by class for easier loading during training
    group_cifar10_by_class(download=False)
    for size in [32, 64, 128]:
        group_imagenette_by_class(size, download=False)
########################################################################
########################################################################


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        choices=['synth', 'real', 'images'])
    args = parser.parse_args()

    if 'synth' in args.datasets:
        create_plot_synth_datasets()

    if 'real' in args.datasets:
        preprocess_singlecell()

    if 'images' in args.datasets:
        preprocess_images()


if __name__ == '__main__':
    main()
