###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Authors: Yulia Rubanova and Ricky Chen
###########################

import os
import argparse
import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt
import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device
from sklearn import model_selection


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


# Adapted from: https://github.com/rtqichen/time-series-datasets


# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max


class PhysioNet(object):

    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root, train=True, download=False, quantization=0.1, n_samples=None, device=torch.device("cpu")):

        self.root = root
        self.train = train
        self.reduce = "average"
        self.quantization = quantization

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, data_file))
            self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]

    def download(self):
        if self._check_exists():
            return

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)

                torch.save(labels, os.path.join(self.processed_folder, filename.split('.')[0] + '.pt'))

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(self.params)).to(self.device)]
                    mask = [torch.zeros(len(self.params)).to(self.device)]
                    nobs = [torch.zeros(len(self.params))]
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        # Time in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        time = round(time / self.quantization) * self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.params)).to(self.device))
                            mask.append(torch.zeros(len(self.params)).to(self.device))
                            nobs.append(torch.zeros(len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            #vals[-1][self.params_dict[param]] = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == 'RecordID', 'Read unexpected param {}'.format(param)
                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # Only training set has labels
                    labels = outcomes[record_id]
                    # Out of 5 label types provided for Physionet, take only the last one -- mortality
                    labels = labels[4]

                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(self.processed_folder,
                             filename.split('.')[0] + "_" + str(self.quantization) + '.pt'))

        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                    os.path.join(self.processed_folder,
                                 filename.split('.')[0] + "_" + str(self.quantization) + '.pt')):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def test_file(self):
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def label_file(self):
        return 'Outcomes-a.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str

    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        non_zero_attributes = (torch.sum(mask, 0) > 2).numpy()
        non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
        n_non_zero = sum(non_zero_attributes)

        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}

        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

        #for i in range(len(self.params)):
        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:, param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.]
            data_cur_param = data[tp_mask == 1., param_id]

            ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(), marker='o')
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)


def variable_time_collate_fn(batch, args, device=torch.device("cpu"), data_type="train", data_min=None, data_max=None):
    """
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device=device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, att_min=data_min, att_max=data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {"data": combined_vals, "time_steps": combined_tt, "mask": combined_mask, "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict


#####################################################################################################
def parse_datasets(args, device):

    def basic_collate_fn(batch, time_steps, args=args, device=device, data_type="train"):
        batch = torch.stack(batch)
        data_dict = {"data": batch, "time_steps": time_steps}

        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type=data_type)
        return data_dict

    dataset_name = args.dataset

    n_total_tp = args.timepoints + args.extrap
    max_t_extrap = args.max_t / args.timepoints * n_total_tp

    # Physionet dataset

    if dataset_name == "physionet":
        train_dataset_obj = PhysioNet(
            'data/physionet',
            train=True,
            quantization=args.quantization,
            download=True,
            n_samples=min(10000, args.n),
            device=device)
        # Use custom collate_fn to combine samples with arbitrary time observations.
        # Returns the dataset along with mask and time steps
        test_dataset_obj = PhysioNet(
            'data/physionet',
            train=False,
            quantization=args.quantization,
            download=True,
            n_samples=min(10000, args.n),
            device=device)

        # Combine and shuffle samples from physionet Train and physionet Test
        total_dataset = train_dataset_obj[:len(train_dataset_obj)]

        if not args.classif:
            # Concatenate samples from original Train and Test sets
            # Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
            total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(
            total_dataset, train_size=0.8, random_state=42, shuffle=True)

        record_id, tt, vals, mask, labels = train_data[0]

        n_samples = len(total_dataset)
        input_dim = vals.size(-1)

        batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
        data_min, data_max = get_data_min_max(total_dataset)

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(
                batch, args, device, data_type="train", data_min=data_min, data_max=data_max))
        test_dataloader = DataLoader(
            test_data,
            batch_size=n_samples,
            shuffle=False,
            collate_fn=lambda batch: variable_time_collate_fn(
                batch, args, device, data_type="test", data_min=data_min, data_max=data_max))

        attr_names = train_dataset_obj.params
        data_objects = {
            "dataset_obj": train_dataset_obj,
            "train_dataloader": utils.inf_generator(train_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "input_dim": input_dim,
            "n_train_batches": len(train_dataloader),
            "n_test_batches": len(test_dataloader),
            "attr": attr_names,    #optional
            "classif_per_tp": False,    #optional
            "n_labels": 1
        }    #optional
    return data_objects


if __name__ == '__main__':
    torch.manual_seed(1991)

    parser = argparse.ArgumentParser('Physionet_preprocessing')

    parser.add_argument('-n', type=int, default=8000, help="Size of the dataset")
    parser.add_argument('-b', '--batch-size', type=int, default=50)
    parser.add_argument(
        '--dataset',
        type=str,
        default='physionet',
        help="Dataset to load. Available: physionet, activity, hopper, periodic")
    parser.add_argument(
        '-s',
        '--sample-tp',
        type=float,
        default=None,
        help="Number of time points to sub-sample."
        "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample"
    )

    parser.add_argument(
        '-c',
        '--cut-tp',
        type=int,
        default=None,
        help="Cut out the section of the timeline of the specified length (in number of points)."
        "Used for periodic function demo.")

    parser.add_argument(
        '--quantization',
        type=float,
        default=0.016,
        help="Quantization on the physionet dataset."
        "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
    parser.add_argument(
        '--classif',
        action='store_true',
        help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
    parser.add_argument(
        '--extrap',
        action='store_true',
        help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
    parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
    parser.add_argument('--max-t', type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_obj = parse_datasets(args, device)
    print(data_obj)
    train_dataloader = data_obj['train_dataloader']
    print("Check Batch dict  ,, batch_dict['observed_data'] ")
    print(
        "Dict_keys = ['observed_data', 'observed_tp', 'data_to_predict', 'tp_to_predict', 'observed_mask', 'mask_predicted_data', 'labels', 'mode'] "
    )
    batch_dict = train_dataloader.__next__()
