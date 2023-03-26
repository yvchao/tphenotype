import abc

import numpy as np
import torch
from tqdm import auto

from .utils.aggregator import MetricAggregator
from .utils.dataset import Dataset
from .utils.utils import calculate_loss, get_summary


class BaseModel(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, train_set):
        pass


class NNBaseModel(BaseModel, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Base Model"
        self.device = "cpu"

    @abc.abstractmethod
    def forward(self, X):
        pass

    @abc.abstractmethod
    def _calculate_train_losses(self, batch):
        pass

    @abc.abstractmethod
    def _calculate_valid_losses(self, batch):
        pass

    def _train_epoch(self, loss_weights, data_loader, optimizer, max_grad_norm=1, **kwargs):
        self.train()
        agg = MetricAggregator()
        for batch in data_loader:
            optimizer.zero_grad()

            batch = self._data_to_device(batch)

            losses = self._calculate_train_losses(batch)

            L = calculate_loss(losses, loss_weights)
            L.to(self.device)

            metrics = {k: v.detach().item() for k, v in losses.items()}
            metrics["total"] = L.detach().cpu().item()
            agg.update(metrics)

            if L.requires_grad:
                L.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()

        metrics_train = agg.query()
        loss_train = metrics_train.pop("total")
        return loss_train, metrics_train

    def _valid_epoch(self, loss_weights, valid_set):
        self.eval()
        if valid_set is not None:
            with torch.no_grad():
                valid_set = self._data_to_device(valid_set)
                losses = self._calculate_valid_losses(valid_set)
                L = calculate_loss(losses, loss_weights)
                loss_valid = L.detach().cpu().item()
                metrics_valid = {k: v.detach().item() for k, v in losses.items()}
        else:
            loss_valid = None
            metrics_valid = {}
        return loss_valid, metrics_valid

    def _clone_state_dict(self):
        state_dict = {}
        for key in self.state_dict():
            state_dict[key] = self.state_dict()[key].clone()
        return state_dict

    def _data_to_device(self, data):
        for k, v in data.items():
            data[k] = v.to(self.device)

        return data

    def _get_optimizer(self, parameters, learning_rate):
        betas = (0.9, 0.9)
        if parameters is not None:
            optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=betas)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas)
        return optimizer

    def _prepare_dataset(self, train_set, valid_set, batch_size):
        train_set = Dataset(train_set)
        train_set = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

        if valid_set is not None:
            valid_set = Dataset(valid_set)
            valid_set = valid_set.get_all_data()
        return train_set, valid_set

    def fit(
        self,
        train_set,
        loss_weights,
        valid_set=None,
        learning_rate=0.1,
        batch_size=50,
        epochs=100,
        max_grad_norm=1,
        tolerance=None,
        parameters=None,
        return_history=False,
        verbose=True,
        **kwargs,
    ):
        loss_history = {}
        loss_history["train"] = np.full(epochs, np.NaN)
        loss_history["valid"] = np.full(epochs, np.NaN)

        train_set, valid_set = self._prepare_dataset(train_set, valid_set, batch_size)

        optimizer = self._get_optimizer(parameters, learning_rate)

        best_validation_loss = torch.inf
        no_improvement_count = 0
        best_model = {}

        with auto.trange(epochs, position=0, leave=True, disable=not verbose) as tbar:
            for epoch in tbar:
                loss_train, metrics_train = self._train_epoch(
                    loss_weights, train_set, optimizer, max_grad_norm=max_grad_norm, **kwargs
                )
                loss_history["train"][epoch] = loss_train

                loss_valid, metrics_valid = self._valid_epoch(loss_weights, valid_set)
                loss_history["valid"][epoch] = loss_valid

                if loss_valid is not None:
                    if best_validation_loss > loss_valid:
                        best_validation_loss = loss_valid
                        no_improvement_count = 0

                        best_model = self._clone_state_dict()
                    else:
                        no_improvement_count += 1

                    if tolerance is not None and no_improvement_count >= tolerance:
                        break
                else:
                    pass

                summary_train = get_summary(metrics_train)
                summary_valid = get_summary(metrics_valid)
                summary = f"tr-{summary_train}|vl-{summary_valid}"
                tbar.set_description(summary)

        if best_model != {}:
            self.load_state_dict(best_model)

        if return_history:
            return self, loss_history
        else:
            return self

    def save(self, path=".", name=None):
        state_dict = self.state_dict()
        save_name = name if name is not None else f"{self.name}.pt"
        torch.save(state_dict, f"{path}/{save_name}")

    def load(self, filename):
        state_dict = torch.load(filename, map_location="cpu")
        self.load_state_dict(state_dict)
        return self.to(self.device)
