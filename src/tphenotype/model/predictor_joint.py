import numpy as np
import torch
from ..utils.utils import check_shape, select_by_steps, EPS
from ..utils.decorators import device_init, numpy_io, run_in_batch
from ..utils.metrics import get_auc_scores
from ..utils.losses import cross_entropy

from .predictor import Predictor


class JointPredictor(Predictor):

    @device_init
    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.name = 'T-Phenotype(joint)'

    @numpy_io
    def predict_proba_g(self, x, t):
        t = check_shape(t).to(self.device)
        x = x.to(self.device)
        with torch.no_grad():
            out = self.forward({'x': x,'t':t})
        return out["prob"]
    

    def forward(self, input):
        # t: batch_size x series_size
        # x: batch_size x series_size x x_dim
        # y: batch_size x series_size x y_dim
        # mask: batch_size x series_size
        t = input['t']
        x = input['x']
        #y = input['y']
        #mask = input['mask']
        x_rep = self._encode(x,t).detach()

        z = self.g(x_rep)
        prob = self._get_probs(z)

        out = {}
        out["prob"] = prob
        return out

    def _calculate_train_losses(self, batch):
        # t: batch_size x series_size
        # x: batch_size x series_size x x_dim
        # y: batch_size x series_size x y_dim
        # mask: batch_size x series_size
        t = batch['t']
        x = batch['x']
        y = batch['y']
        mask = batch['mask']

        out = self.forward(batch)
        y_pred = out['prob']

        losses = {}
        losses['ce'] = cross_entropy(y_pred, y, mask=mask[:, :])
        
        for i, d in enumerate(self.time_series_dims):
            x_d = x[:, :, d]
            encoder_losses = self.encoders[i].expose_loss(x_d,t,mask)
            for k,v in encoder_losses.items():
                losses[k] = losses.get(k,0.0) + v / len(self.time_series_dims)
        return losses

    def _calculate_valid_losses(self, batch):
        t = batch['t']
        x = batch['x']
        mask = batch['mask']
        y = batch['y'].cpu()

        out = self.forward(batch)
        y_pred = out['prob'].detach().cpu()

        losses = {}
        losses['ce'] = cross_entropy(y_pred, y, mask=mask.cpu())

        for i,d in enumerate(self.time_series_dims):
            x_d = x[:,:,d]
            encoder_losses = self.encoders[i].expose_loss(x_d,t,mask)
            losses['rmse'] = losses.get('rmse',0.0) + encoder_losses['rmse'].detach().cpu() / len(self.time_series_dims)
        #

        AUROC, AUPRC = get_auc_scores(y, y_pred, mask=mask.cpu())
        losses['AUROC'] = torch.tensor(np.mean(AUROC))
        losses['AUPRC'] = torch.tensor(np.mean(AUPRC))

        return losses


    def fit(self,
            train_set,
            loss_weights,
            valid_set=None,
            learning_rate=0.1,
            batch_size=50,
            epochs=100,
            max_grad_norm=1,
            tolerance=None,
            device=None,
            parameters=None,
            verbose=True,
            **kwargs):

        args = locals().copy()
        args.pop('self')
        super(Predictor, self).fit(**args)

        x = train_set['x']
        t = train_set['t']
        mask = train_set['mask']

        self.cls.verbose = verbose
        if self.cls.K>0:
            self.cls.fit(x, t, mask)
        if verbose:
            print(f'done')

        return self
