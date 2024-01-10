import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class CSTLoss(nn.Module):
    def __init__(self, alpha=2, reduction="mean", base_loss=F.binary_cross_entropy,
                 s_loss=F.kl_div):
        super(CSTLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.base_loss = base_loss
        self.s_loss = s_loss

    def forward(self, inputs, dist_inputs, targets):
        l_0 = self.base_loss(inputs, targets, reduction=self.reduction)
        l_dist_1 = self.s_loss(
            torch.log(torch.clip(inputs + 1e-7, 0., 1.)),
            torch.clip(dist_inputs + 1e-7, 0., 1.),
            reduction=self.reduction
        )
        l_dist_2 = F.kl_div(
            torch.log(torch.clip(1 - inputs + 1e-7, 0., 1.)),
            torch.clip(1 - dist_inputs + 1e-7, 0., 1.),
            reduction=self.reduction
        )
        return  l_0 + self.alpha * (l_dist_1 + l_dist_2)


def train_cst(
    model,
    train_loader,
    # device: torch.device,
    device,
    optimizer,
    dist_layer,
    val_loader=None,
    alpha=2,
    base_loss=F.binary_cross_entropy,
    s_loss=F.kl_div,
    epochs: int = 1,
    model_save_path: str = "",
    model_base_name: str = "new_model"

):
    loss_fn = CSTLoss(alpha=alpha, base_loss=base_loss, s_loss=s_loss)
    model.to(device)
    if epochs < 1: epochs = 1
    metrics = pd.DataFrame(columns=["epoch", "cst_loss", "l0", "l_stab", "val_l0"])
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()  # sets the model layers to training mode
        cst_loss_train, l0_train, l_stab_train = [], [], []
        pbar = tqdm(train_loader)
        ## TRAIN
        for i, data in enumerate(pbar, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            dist_inputs = dist_layer(inputs.type(torch.uint8)).float().to(device)
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)['out']
            dist_outputs = model(dist_inputs)['out']
            l0 = base_loss(outputs, labels)
            l_stab = s_loss(torch.log(torch.clip(outputs + 1e-7, 0., 1.)),
                            torch.clip(dist_outputs + 1e-7, 0., 1.),)
            cst_loss = loss_fn(outputs, dist_outputs, labels)
            cst_loss.backward()
            optimizer.step()

            ## print statistics
            # moving average of 10 batches is shown for less jumpy results
            cst_loss_train.append(cst_loss.item())
            l0_train.append(l0.item())
            l_stab_train.append(l_stab.item())
            # if len(cst_loss_train) > 10:
            #     cst_loss_train.pop(0)
            #     l0_train.pop(0)
            #     l_stab_train.pop(0)

            cst_loss_mean = np.array(cst_loss_train).mean()
            l0_mean = np.array(l0_train).mean()
            l_stab_mean = np.array(l_stab_train).mean()

            pbar_desc = f"Epoch {epoch} train - batch {i+1}/{len(train_loader)}" \
                        f" - cst_loss: {cst_loss_mean:.3f}" \
                        f" - l0: {l0_mean:.3f}"\
                        f" - lstab: {l_stab_mean:.3f}"
            pbar.set_description(pbar_desc)

        ## VALIDATION
        if val_loader is not None:
            model.eval()   # sets layers to eval mode

            pbar = tqdm(val_loader)
            cst_loss_val, l0_val, l_stab_val = [], [], []
            for i, data in enumerate(pbar, 0):
                inputs, labels = data
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)
                outputs = model(inputs)['out']

                l0 = base_loss(outputs, labels)

                ## print statistics
                # moving average of 10 batches is shown for less jumpy results
                # cst_loss_val.append(cst_loss.item())
                l0_val.append(l0.item())
                # l_stab_val.append(l_stab.item())
                # if len(cst_loss_val) > 10:
                #     # cst_loss_val.pop(0)
                #     l0_val.pop(0)
                #     # l_stab_val.pop(0)

                l0_val_mean = np.array(l0_val).mean()

                pbar_desc = f"Epoch {epoch} val - batch {i+1}/{len(val_loader)}" \
                            f" - val_l0: {l0_val_mean:.3f}" \
                            # f" - cst_loss: {np.array(cst_loss_val).mean():.3f}" \
                            # f" - lstab: {np.array(l_stab_val).mean():.3f}"
                pbar.set_description(pbar_desc)

        # save model epoch
        new_path = Path(model_save_path) / model_base_name
        new_path.mkdir(parents=True, exist_ok=True)
        path = new_path / (model_base_name + f"_e{epoch}.pt")
        torch.save(model.state_dict(), path)

        # save metrics
        row = {"epoch": [epoch], "cst_loss": [cst_loss_mean], "l0": [l0_mean],
               "l_stab": [l_stab_mean], "val_l0": [l0_val_mean]}
        metrics = pd.concat([metrics, pd.DataFrame(row)])
        metrics.to_csv(new_path / (model_base_name + ".csv"))
