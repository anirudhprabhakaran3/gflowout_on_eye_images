import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import models


import random
import numpy as np

from utils.GFNUtils import (
    RandomMaskGenerator,
    CNN_MLP,
    construct_conditional_mask_generators,
    construct_multiinput_conditional_mask_generators,
    construct_unconditional_mask_generators,
)
from utils.options import Options

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)


class ViT_GFN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        activation=nn.LeakyReLU,
        opt=None,
    ):
        super().__init__()

        self.opt = opt
        self.num_classes = num_classes
        self.beta = 1
        self.training = True

        self.vit = models.vit_b_16()
        self.vit.load_state_dict(torch.load("vit_b_16-c867db91.pth"))
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, self.num_classes), nn.Softmax(dim=1)
        )

        self.random_chance = 0.5
        self.temperature = 2
        self.maskgenerator_input_shapes = [(197, 768)] * 12

        self.rand_mask_generator = RandomMaskGenerator(dropout_rate=opt.mlp_dr)
        hiddens = [32, 32]

        self.p_zx_mask_generators = construct_conditional_mask_generators(
            n_channels=[DIMs[0] for DIMs in self.maskgenerator_input_shapes],
            layer_dims=[DIMs[0] * DIMs[1] for DIMs in self.maskgenerator_input_shapes],
            additional_input_dims=[0 for j in self.maskgenerator_input_shapes],
            hiddens=hiddens,
        ).to(device)

        self.q_zxy_mask_generators = construct_multiinput_conditional_mask_generators(
            n_channels=[DIMs[0] for DIMs in self.maskgenerator_input_shapes],
            layer_dims=[DIMs[0] * DIMs[1] for DIMs in self.maskgenerator_input_shapes],
            additional_input_dims=[
                num_classes for j in self.maskgenerator_input_shapes
            ],
            hiddens=hiddens,
        ).to(device)

        self.p_z_mask_generators = RandomMaskGenerator(dropout_rate=opt.mlp_dr)
        self.q_z_mask_generators = construct_unconditional_mask_generators(
            layer_dims=[x[0] for x in self.maskgenerator_input_shapes], hiddens=hiddens
        ).to(device)

        self.activation = activation

        self.LogZ_total_flowestimator = CNN_MLP(
            cnn_in_dim=(3, 224, 224),
            mlp_in_dim=num_classes,
            out_dim=1,
            activation=nn.LeakyReLU,
        ).to(device)
        self.LogZ_unconditional = nn.Parameter(torch.tensor(0.0))

        z_lr = 1e-1
        mg_lr_mu = 1e-3
        lr = opt.lr
        self.beta = 1
        self.elbo = torch.zeros(1)
        self.N = 0

        q_z_param_list = [
            {
                "params": self.q_z_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            },
            {"params": self.LogZ_unconditional, "lr": z_lr, "weight_decay": 0.1},
        ]
        self.q_z_optimizer = optim.Adam(q_z_param_list, lr=opt.lr)

        p_zx_param_list = [
            {
                "params": self.p_zx_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            },
        ]
        self.p_zx_optimizer = optim.Adam(p_zx_param_list, lr=opt.lr)

        q_zxy_param_list = [
            {
                "params": self.q_zxy_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            },
            {
                "params": self.LogZ_total_flowestimator.parameters(),
                "lr": z_lr,
                "weight_decay": 0.1,
            },
        ]
        self.q_zxy_optimier = optim.Adam(q_zxy_param_list, lr=opt.lr)

        taskmodel_param_list = [{"params": self.vit.heads.parameters(), "lr": lr}]
        self.taskmodel_optimizer = optim.SGD(
            taskmodel_param_list, momentum=0.9, weight_decay=5e-4, lr=opt.lr
        )

        self.taskmodel_scheduler = optim.lr_scheduler.MultiStepLR(
            self.taskmodel_optimizer, milestones=self.opt.schedule_milestones, gamma=0.1
        )

        self.to(device)

    def gfn_forward(self, x, y, mask="none"):
        y = F.one_hot(y, self.num_classes).float()
        batch_size = x.shape[0]

        LogZ_unconditional = self.LogZ_unconditional
        LogZ_conditional = self.LogZ_total_flowestimator(x, y)

        LogPF_qz = torch.zeros(batch_size).to(device)
        LogPB_qz = torch.zeros(batch_size).to(device)
        LogR_qz = torch.zeros(batch_size).to(device)
        LogPF_BNN = torch.zeros(batch_size).to(device)
        LogPF_qzxy = torch.zeros(batch_size).to(device)
        LogPB_qzxy = torch.zeros(batch_size).to(device)
        LogR_qzxy = torch.zeros(batch_size).to(device)
        Log_pzx = torch.zeros(batch_size).to(device)
        Log_pz = torch.zeros(batch_size).to(device)

        masks_qz = [[] for _ in range(len(self.maskgenerator_input_shapes))]

        for layer_idx in range(len(self.maskgenerator_input_shapes)):
            if mask == "topdown":
                EPSILON = random.uniform(0, 1)
                if layer_idx == 0:
                    if (EPSILON < self.random_chance) and self.training:
                        qz_mask_l = self.rand_mask_generator(
                            torch.zeros(
                                batch_size, self.maskgenerator_input_shapes[0][0]
                            ).to(device)
                        ).to(device)
                    else:
                        qz_mask_l = self.q_z_mask_generators[0](
                            torch.zeros(batch_size, 48400).to(device), self.temperature
                        )
                    qz_p_l = self.q_z_mask_generators[layer_idx].prob(
                        torch.zeros(batch_size, 48400).to(device), qz_mask_l
                    )
                else:
                    previous_mask = []
                    for j in range(layer_idx):
                        previous_mask.append(masks_qz[j][-1])
                    previous_mask = torch.cat(previous_mask, 1)

                    if (EPSILON < self.random_chance) and self.training:
                        qz_mask_l = self.rand_mask_generator(previous_mask).to(device)
                    else:
                        qz_mask_l = self.q_z_mask_generators[layer_idx](
                            previous_mask
                        ).to(device)
                    qz_p_l = self.q_z_mask_generators[layer_idx].prob(
                        previous_mask.to(device), qz_mask_l.to(device)
                    )

                masks_qz[layer_idx].append(qz_mask_l.detach())
                LogPF_qz += (
                    qz_mask_l * torch.log(qz_p_l)
                    + (1.0 - qz_mask_l) * torch.log(1.0 - qz_p_l)
                ).sum(1)
                LogPB_qz += 0
            else:
                masks_qz[layer_idx].append(
                    torch.ones(self.maskgenerator_input_shapes[layer_idx][0]).to(device)
                )

        actual_masks = []
        masks_conditional = []

        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        for layer_idx, layer in enumerate(self.vit.encoder.layers):
            out = layer(x)

            EPSILON = random.uniform(0, 1)

            if mask == "bottomup":
                if self.training:
                    if layer_idx == 0:
                        if EPSILON > self.random_chance:
                            m_conditional_l = self.q_zxy_mask_generators[layer_idx](
                                torch.zeros(batch_size, out.shape[1]).to(device),
                                out.reshape(batch_size, -1).detach(),
                                y.float(),
                            )
                        else:
                            m_conditional_l = self.rand_mask_generator(
                                torch.zeros(out.shape[0], out.shape[1])
                            ).to(device)
                        qzxy_p_l = self.q_zxy_mask_generators[layer_idx].prob(
                            torch.zeros(batch_size, out.shape[1]).to(device),
                            out.reshape(batch_size, -1).detach(),
                            y.float(),
                            m_conditional_l,
                        )
                    else:
                        previous_actual_mask = []
                        for j in range(layer_idx):
                            previous_actual_mask.append(actual_masks[j])
                        previous_actual_mask = torch.cat(previous_actual_mask, 1)

                        if EPSILON > self.random_chance:
                            m_conditional_l = self.q_zxy_mask_generators[layer_idx](
                                previous_actual_mask,
                                out.reshape(batch_size, -1).detach(),
                                y.float(),
                            )
                        else:
                            m_conditional_l = self.rand_mask_generator(
                                torch.zeros(out.shape[0], out.shape[1])
                            ).to(device)
                        qzxy_p_l = self.q_zxy_mask_generators[layer_idx].prob(
                            previous_actual_mask,
                            out.reshape(batch_size, -1).detach(),
                            y.detach().float(),
                            m_conditional_l,
                        )

                    masks_conditional.append(m_conditional_l)
                    LogPF_qzxy += (
                        m_conditional_l * torch.log(qzxy_p_l)
                        + (1.0 - m_conditional_l) * torch.log(1.0 - qzxy_p_l)
                    ).sum(1)
                    LogPB_qzxy -= 0
                else:
                    if layer_idx == 0:
                        m_conditional_l = self.p_zx_mask_generators[layer_idx](
                            out.detach().reshape(out.shape[0], -1)
                        )
                    else:
                        previous_actual_mask = []
                        for j in range(layer_idx):
                            previous_actual_mask.append(actual_masks[j])
                        input_pzx = torch.cat(
                            previous_actual_mask
                            + [out.detach().reshape(out.shape[0], -1)],
                            1,
                        )
                        m_conditional_l = self.p_zx_mask_generators[layer_idx](
                            input_pzx
                        )
                    masks_conditional.append(m_conditional_l)
            else:
                masks_conditional.append(torch.ones(out.shape).to(device))

            if mask == "random":
                m = self.rand_mask_generator(
                    torch.zeros(out.shape[0], out.shape[1])
                ).to(device)
            elif mask == "topdown":
                m = masks_qz[layer_idx][-1]
            elif mask == "bottomup":
                m = m_conditional_l
            elif mask == "none":
                m = torch.ones(out.shape[0], out.shape[1]).to(device)

            m = m.detach()

            if layer_idx == 0:
                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(
                    out.reshape(batch_size, -1).detach(), m
                )
                Log_P_z_l = self.p_z_mask_generators.log_prob(m, m)
            else:
                previous_actual_mask = []
                for j in range(layer_idx):
                    previous_actual_mask.append(actual_masks[j])

                input_pzx = torch.cat(
                    previous_actual_mask + [out.reshape(batch_size, -1).detach()],
                    1,
                )
                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(input_pzx, m)
                Log_P_z_l = self.p_z_mask_generators.log_prob(m, m)
            Log_pzx += Log_P_zx_l
            Log_pz += Log_P_z_l

            x = out
            actual_masks.append(m)
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        x = x[:, 0]
        pred = self.vit.heads(x)

        return (
            pred,
            actual_masks,
            masks_qz,
            masks_conditional,
            LogZ_unconditional,
            LogPF_qz,
            LogR_qz,
            LogPB_qz,
            LogPF_BNN,
            LogZ_conditional,
            LogPF_qzxy,
            LogR_qzxy,
            LogPB_qzxy,
            Log_pzx,
            Log_pz,
        )

    def _gfn_step(self, x, y, mask="none"):
        metric = {}
        (
            logits,
            actual_masks,
            _,
            _,
            LogZ_unconditional,
            LogPF_qz,
            LogR_qz,
            LogPB_qz,
            LogPF_BNN,
            LogZ_conditional,
            LogPF_qzxy,
            LogR_qzxy,
            LogPB_qzxy,
            Log_pzx,
            Log_pz,
        ) = self.gfn_forward(x, y, mask)

        CELoss = nn.CrossEntropyLoss()(logits, y)
        LL = -CELoss

        LogR_unconditional = self.beta * self.N * LL.detach() + Log_pz.detach()
        GFN_loss_unconditional = (
            LogZ_unconditional + LogPF_qz - LogR_unconditional - LogPB_qz
        ) ** 2

        LogR_conditional = self.beta * LL.detach() + Log_pzx.detach()
        GFN_loss_conditional = (
            LogZ_conditional + LogPF_qzxy - LogR_conditional - LogPB_qzxy
        ) ** 2

        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)

        metric["CELoss"] = CELoss.mean().item()
        metric["GFN_loss_unconditional"] = GFN_loss_unconditional.mean().item()
        metric["GFN_loss_conditional"] = GFN_loss_conditional.mean().item()
        metric["acc"] = acc

        COR_qz = np.corrcoef(
            LogPF_qz.detach().cpu().numpy(), LogR_unconditional.detach().cpu()
        )[0, 1]

        COR_qzxy = np.corrcoef(
            LogPF_qzxy.cpu().detach().numpy(), LogR_conditional.cpu().detach().numpy()
        )[0, 1]
        metric["COR_qz"] = COR_qz
        metric["COR_qzxy"] = COR_qzxy

        if mask == "topdown":
            self.q_z_optimizer.zero_grad()
            GFN_loss_unconditional.mean().backward()
            CELoss.mean().backward()
            self.q_z_optimizer.step()
        elif mask == "bottomup":
            self.q_zxy_optimier.zero_grad()
            self.p_zx_optimizer.zero_grad()

            GFN_loss_conditional.mean().backward()
            CELoss.mean().backward()
            Log_pzx.mean().backward()

            self.q_zxy_optimier.step()
            self.p_zx_optimizer.step()
        else:
            CELoss.mean().backward()

        n_units = 0
        n_dropped = 0
        for layer_idx in range(len(self.maskgenerator_input_shapes)):
            m = actual_masks[layer_idx]
            n_units += m.shape[1]
            n_dropped += (m == 0).float().mean(0).sum()

        metric["actual_dropout_rate"] = (n_dropped / n_units).cpu()
        metric["LogZ_unconditional"] = LogZ_unconditional.mean().item()
        metric["LogPF_qz"] = LogPF_qz.mean().item()
        metric["LogR_qz"] = LogR_qz.mean().item()
        metric["LogPB_qz"] = LogPB_qz.mean().item()
        metric["Log_pz"] = Log_pz.mean().item()
        metric["LogZ_conditional"] = LogZ_conditional.mean().item()
        metric["LogPF_qzxy"] = LogPF_qzxy.mean().item()
        metric["LogR_qzxy"] = LogR_qzxy.mean().item()
        metric["LogPB_qzxy"] = LogPB_qzxy.mean().item()
        metric["Log_pzx"] = Log_pzx.mean().item()
        metric["LogPF_BNN"] = LogPF_BNN.mean().item()

        return metric

    def forward(self, x, y, mask="topdown"):
        if self.training:
            (
                logits,
                actual_masks,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self.gfn_forward(x, y, mask)
        else:
            logits = []
            (
                logits_,
                actual_masks,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self.gfn_forward(x, y, mask)
            logits.append(logits_.unsqueeze(2))
            logits = torch.logsumexp(torch.cat(logits, 2), 2)

        return logits, actual_masks

