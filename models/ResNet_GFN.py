import torch
import torch.nn as nn
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
from utils.config import config


class ResNetGFN(nn.Module):
    def __init__(self, activation=nn.LeakyReLU, num_classes=2):
        super(ResNetGFN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.training = config.gfn_model_training

        if config.use_pretrained:
            self.resnet = models.resnet18(pretrained=True, num_classes=1000)
            self.resnet.fc = nn.Linear(512, num_classes)
        else:
            self.resnet = models.resnet18(pretrained=False, num_classes=num_classes)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()
        self.to(self.device)

        ### GFN Related ###
        # Code adapted from Dianbo Liu.
        # https://github.com/kaiyuanmifen/GFNDropout

        self.random_chance = 0
        self.temperature = 2
        self.N = 0

        self.mask_generator_input_shapes = [
            (64, 224, 224),
            (64, 224, 224),
            (128, 112, 112),
            (128, 112, 112),
            (256, 56, 56),
            (256, 56, 56),
            (512, 28, 28),
            (512, 28, 28),
            # (64, 32, 32),
            # (64, 32, 32),
            # (128, 16, 16),
            # (128, 16, 16),
            # (256, 8, 8),
            # (258, 8, 8),
            # (512, 4, 4),
            # (512, 4, 4),
        ]

        hiddens = [32, 32]
        self.random_mask_generator = RandomMaskGenerator(
            dropout_rate=config.mlp_dropout_rate
        )

        self.p_zx_mask_generators = construct_conditional_mask_generators(
            n_channels=[DIMs[0] for DIMs in self.mask_generator_input_shapes],
            layer_dims=[
                DIMs[0] * DIMs[1] * DIMs[2] for DIMs in self.mask_generator_input_shapes
            ],
            additional_input_dims=[0 for j in self.mask_generator_input_shapes],
            hiddens=hiddens,
        )

        self.q_zxy_mask_generators = construct_multiinput_conditional_mask_generators(
            n_channels=[DIMs[0] for DIMs in self.mask_generator_input_shapes],
            layer_dims=[
                DIMs[0] * DIMs[1] * DIMs[2] for DIMs in self.mask_generator_input_shapes
            ],
            additional_input_dims=[
                num_classes for j in self.mask_generator_input_shapes
            ],
            hiddens=[32, 32],
        ).to(self.device)

        self.p_z_mask_generators = RandomMaskGenerator(
            dropout_rate=config.mlp_dropout_rate
        )
        self.q_z_mask_generators = construct_unconditional_mask_generators(
            layer_dims=[x[0] for x in self.mask_generator_input_shapes], hiddens=hiddens
        ).to(self.device)

        self.activation = activation
        mg_activation = nn.LeakyReLU

        self.logz_total_flowestimator = CNN_MLP(
            CNN_in_dim=(3, 224, 224),
            mlp_in_dim=num_classes,
            out_dim=1,
            activation=mg_activation,
        ).to(self.device)

        self.logz_unconditional = nn.Parameter(torch.tensor(0.0))

        z_lr = 1e-1
        mg_lr_mu = 1e-3
        lr = config.lr
        self.beta = 1  # Temperature on rewards

        q_z_param_list = [
            {
                "params": self.q_z_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            },
            {"params": self.logz_unconditional, "lr": z_lr, "weight_decay": 0.1},
        ]
        self.q_z_optimizer = optim.Adam(q_z_param_list)

        p_zx_param_list = [
            {
                "params": self.p_zx_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            }
        ]
        self.p_zx_optimizer = optim.Adam(p_zx_param_list)

        q_zxy_param_list = [
            {
                "params": self.q_zxy_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            },
            {
                "params": self.logz_total_flowestimator.parameters(),
                "lr": z_lr,
                "weight_decay": 0.1,
            },
        ]
        self.q_zxy_optimizer = optim.Adam(q_zxy_param_list)

        if config.tune_last_layer_only:
            task_model_param_list = [{"params": self.resnet.fc.parameters(), "lr": lr}]
        else:
            task_model_param_list = [{"params": self.resnet.parameters(), "lr": lr}]
        self.task_model_optimizer = optim.SGD(
            task_model_param_list, momentum=0.9, weight_decay=5e-4
        )
        self.task_model_scheduler = optim.lr_scheduler.MultiStepLR(
            self.task_model_optimizer, milestones=config.schedule_milestone, gamma=0.1
        )
        self.to(self.device)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, y, mask="none"):
        x, y = x.to(self.device), y.to(self.device)

        # using GFlowNet
        if self.training:
            (
                logits,
                actual_masks,
                masks_qz,
                masks_qzxy,
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
            ) = self.GFN_forward(x, y, mask)
        else:
            # The inference code is already set certain number of repeats, so set to 1 here
            # Sample multiple times and use average as inference prediction because GFN cannot take expectation easily
            N_repeats = 1
            logits = []
            for _ in range(N_repeats):
                (
                    logits_,
                    actual_masks,
                    masks_qz,
                    masks_qzxy,
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
                ) = self.GFN_forward(x, y, mask)
                logits.append(logits_.unsqueeze(2))
            logits = torch.logsumexp(torch.cat(logits, 2), 2)
        return logits, actual_masks

    def GFN_forward(self, x, y, mask="none"):
        y = nn.functional.one_hot(y, self.num_classes).float()
        batch_size, input_dim = x.shape[0], x.shape[1]
        logz_unconditional = (
            self.logz_unconditional
        )  # Top Down mask has partition function independent of input x
        logz_conditional = self.logz_total_flowestimator(
            x, y
        )  # Bottom up mask has partition function dependent on input x

        logpf_qz = torch.zeros(batch_size).to(self.device)  # Forward probability
        logpb_qz = torch.zeros(batch_size).to(self.device)  # Backward probability
        logr_qz = torch.zeros(batch_size).to(self.device)  # Reward

        logpf_bnn = torch.zeros(batch_size).to(self.device)
        logpb_bnn = torch.zeros(batch_size).to(self.device)

        logpf_qzxy = torch.zeros(batch_size).to(
            self.device
        )  # Forward probability, for conditional mask generator
        logpb_qzxy = torch.zeros(batch_size).to(self.device)
        logr_qzxy = torch.zeros(batch_size).to(self.device)

        log_pzx = torch.zeros(batch_size).to(self.device)
        log_pz = torch.zeros(batch_size).to(self.device)

        """ mu Mask Generation, independent of input x """

        # Initialize masks as all zeros (dropout them all)
        # One batch share the same mu mask

        if self.training:
            temperature = self.temperature
        else:
            temperature = 1.0
        masks_qz = [[] for _ in range(len(self.mask_generator_input_shapes))]

        for layer_idx in range(len(self.mask_generator_input_shapes)):
            if mask == "topdown":
                EPSILON = random.uniform(0, 1)
                qz_p_l = None
                if layer_idx == 0:
                    if (EPSILON < self.random_chance) and (self.training):
                        qz_mask_l = self.random_mask_generator(
                            torch.zeros(
                                batch_size, self.mask_generator_input_shapes[0][0]
                            ).to(self.device)
                        ).to(self.device)
                    else:
                        qz_mask_l = self.q_z_mask_generators[layer_idx](
                            torch.zeros(batch_size, 784).to(self.device), temperature
                        )  # 784 is an arbitrary number here.
                    qz_p_l = self.q_z_mask_generators[layer_idx].prob(
                        torch.zeros(batch_size, 784).to(self.device), qz_mask_l
                    )
                else:
                    # Concatenate all previous masks
                    previous_mask = []
                    for j in range(layer_idx):
                        previous_mask.append(masks_qz[j][-1])
                    previous_mask = torch.cat(previous_mask, 1)

                    # During random, random action + tempered policy is used
                    if (EPSILON < self.random_chance) and (self.training):
                        qz_mask_l = self.random_mask_generator(previous_mask).to(
                            self.device
                        )
                    else:
                        qz_mask_l = self.q_z_mask_generators[layer_idx](
                            previous_mask, temperature
                        ).to(self.device)
                    qz_p_l = self.q_z_mask_generators[layer_idx].prob(
                        previous_mask.to(self.device), qz_mask_l.to(self.device)
                    )

                masks_qz[layer_idx].append(qz_mask_l.detach().clone())
                logpf_qz += (
                    qz_mask_l * torch.log(qz_p_l)
                    + (1 - qz_mask_l) * torch.log(1 - qz_p_l)
                ).sum(1)
                logpb_qz += 0  # Uniform backward P
            else:
                masks_qz[layer_idx].append(
                    torch.ones(self.mask_generator_input_shapes[layer_idx][0]).to(
                        self.device
                    )
                )

        # Forward pass
        actual_masks = []
        masks_conditional = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        block_idx = 0
        layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        ]

        for layer in layers:
            for blockid in range(2):
                identity = x
                out = layer[blockid].conv1(x)
                out = layer[blockid].bn1(out)
                out = layer[blockid].relu(out)

                out = layer[blockid].conv2(out)
                out = layer[blockid].bn2(out)

                if layer[blockid].downsample is not None:
                    identity = layer[blockid].downsample(x)

                # Different masks generator
                if block_idx >= 0:  # This implementation has all layers as dropout
                    # For using random masks
                    EPSILON = random.uniform(0, 1)
                    layer_idx = block_idx

                    if "bottomup" in mask:
                        if self.training:
                            # During training, use q(z|x, y:phi) tto sample mask
                            if layer_idx == 0:
                                if EPSILON >= self.random_chance:
                                    # Generate mask based on activation from previous layer, detach from BNN training
                                    m_conditional_l = self.q_zxy_mask_generators[
                                        layer_idx
                                    ](
                                        torch.zeros(batch_size, out.shape[1]).to(
                                            self.device
                                        ),
                                        out.reshape(batch_size, -1).clone().detach(),
                                        y.float().clone().detach(),
                                        temperature,
                                    )
                                else:
                                    m_conditional_l = self.random_mask_generator(
                                        torch.zeros(out.shape[0], out.shape[1])
                                    ).to(self.device)
                                qzxy_p_l = self.q_zxy_mask_generators[layer_idx].prob(
                                    torch.zeros(batch_size, out.shape[1]).to(
                                        self.device
                                    ),
                                    out.reshape(batch_size, -1).clone().detach(),
                                    y.float().clone().detach(),
                                    m_conditional_l,
                                )
                            else:
                                previous_actual_mask = []  # Use previous actual masks
                                for j in range(layer_idx):
                                    previous_actual_mask.append(actual_masks[j])
                                previous_actual_mask = torch.cat(
                                    previous_actual_mask, 1
                                )

                                if EPSILON >= self.random_chance:
                                    m_conditional_l = self.q_zxy_mask_generators[
                                        layer_idx
                                    ](
                                        previous_actual_mask,
                                        out.reshape(batch_size, -1).clone().detach(),
                                        y.float().clone().detach(),
                                        temperature,
                                    )
                                else:
                                    # During training of a certain chance, a random policy will be used to explore the space
                                    m_conditional_l = self.random_mask_generator(
                                        torch.zeros(out.shape[0], out.shape[1])
                                    ).to(self.device)
                                qzxy_p_l = self.q_zxy_mask_generators[layer_idx].prob(
                                    previous_actual_mask,
                                    out.reshape(batch_size, -1).clone().detach(),
                                    y.float().detach().clone(),
                                    m_conditional_l,
                                )

                            masks_conditional.append(m_conditional_l)

                            # Add log P_F_Z to the GFN loss
                            logpf_qzxy += (
                                m_conditional_l * torch.log(qzxy_p_l)
                                + (1 - m_conditional_l) * torch.log(1 - qzxy_p_l)
                            ).sum(1)
                            logpb_qzxy += 0
                        else:
                            # During inference, use p(z|x;xi) to sample mask
                            if layer_idx == 0:
                                m_conditional_l = self.p_zx_mask_generators[layer_idx](
                                    out.clone().detach().reshape(out.shape[0], -1)
                                )
                            else:
                                previous_actual_mask = []
                                for j in range(layer_idx):
                                    previous_actual_mask.append(actual_masks[j])
                                # Calculate p(z|x;xi)
                                input_pzx = torch.cat(
                                    previous_actual_mask
                                    + [out.clone().detach().reshape(out.shape[0], -1)],
                                    1,
                                )
                                # Generate mask based on activation from preious layer, detach from BNN training
                                m_conditional_l = self.p_zx_mask_generators[layer_idx](
                                    input_pzx
                                )
                            masks_conditional.append(m_conditional_l)
                    else:
                        masks_conditional.append(torch.ones(out.shape).to(self.device))

                    if mask == "random":
                        EPSILON = random.uniform(0, 1)
                        m = self.random_mask_generator(
                            torch.zeros(out.shape[0], out.shape[1])
                        ).to(self.device)
                    elif mask == "topdown":
                        m = masks_qz[layer_idx][-1]
                    elif mask == "bottomup":
                        m = m_conditional_l
                    elif mask == "none":
                        m = torch.ones(out.shape[0], out.shape[1]).to(self.device)

                    # Mask shuld not interfere with the backbone model (MLP, ResNet, etc.) training
                    m = m.detach().clone()

                    # Calculate p(z; xi) and p(z|x;xi) both of which are needed for training
                    if layer_idx == 0:
                        # Calculate p(z|x;xi)
                        log_p_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(
                            out.reshape(batch_size, -1).clone().detach(), m
                        )

                        # Calculate p(z|xi)
                        log_p_z_l = self.p_z_mask_generators.log_prob(m, m)
                    else:
                        # Use previous masks
                        previous_actual_mask = []
                        for j in range(layer_idx):
                            previous_actual_mask.append(actual_masks[j])

                        # Calculate p(z|x;xi)
                        input_pzx = torch.cat(
                            previous_actual_mask
                            + [out.reshape(batch_size, -1).clone().detach()],
                            1,
                        )
                        # Generate mask based on activation from previous layer, detach from BNN training.
                        log_p_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(
                            input_pzx, m
                        )

                        # Calculate p(z;xi)
                        # Generate mask based on activation from previous layer, detach from BNN training.
                        log_p_z_l = self.p_z_mask_generators.log_prob(m, m)

                    log_pzx += log_p_zx_l
                    log_pz += log_p_z_l

                    # Apply the mask
                    out = out.mul(m.unsqueeze(2).unsqueeze(3))
                    actual_masks.append(m)

                out += identity
                out = layer[blockid].relu(out)
                x = out

                block_idx += 1

        # FC layer
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        pred = x

        return (
            pred,
            actual_masks,
            masks_qz,
            masks_conditional,
            logz_unconditional,
            logpf_qz,
            logr_qz,
            logpb_qz,
            logpf_bnn,
            logz_conditional,
            logpf_qzxy,
            logr_qzxy,
            logpb_qzxy,
            log_pzx,
            log_pz,
        )

    def _gfn_step(self, x, y, mask_train="", mask="none"):
        # This step allows us to use different x, y to generate masks and calculate rewards (loss)
        metric = {}
        (
            logits,
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
        ) = self.GFN_forward(x, y, mask)

        # Loss Calculation
        CELoss = nn.CrossEntropyLoss(reduction="none")(logits, y)
        LL = -CELoss

        LogR_unconditional = (
            self.beta * self.N * LL.detach().clone() + Log_pz.detach().clone()
        )
        GFN_loss_unconditional = (
            LogZ_unconditional + LogPF_qz - LogR_unconditional - LogPB_qz
        ) ** 2  # +kl_weight*kl#jointly train the last layer BNN and the mask generator

        LogR_conditional = self.beta * LL.detach().clone() + Log_pzx.detach().clone()
        GFN_loss_conditional = (
            LogZ_conditional + LogPF_qzxy - LogR_conditional - LogPB_qzxy
        ) ** 2  # +kl_weight*kl#jointly train the last layer BNN and the mask generator

        # Update the model
        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)

        metric["CELoss"] = CELoss.mean().item()
        metric["GFN_loss_unconditional"] = GFN_loss_unconditional.mean().item()
        metric["GFN_loss_conditional"] = GFN_loss_conditional.mean().item()
        metric["acc"] = acc

        COR_qz = np.corrcoef(
            LogPF_qz.cpu().detach().numpy(), LogR_unconditional.cpu().detach().numpy()
        )[0, 1]
        COR_qzxy = np.corrcoef(
            LogPF_qzxy.cpu().detach().numpy(), LogR_conditional.cpu().detach().numpy()
        )[0, 1]

        metric["COR_qz"] = COR_qz
        metric["COR_qzxy"] = COR_qzxy

        if mask == "topdown":
            # Train q(z) and logZ by GFN loss
            self.q_z_optimizer.zero_grad()
            GFN_loss_unconditional.mean().backward(retain_graph=True)
            self.task_model_optimizer.zero_grad()
            task_model_loss = CELoss
            task_model_loss.mean().backward(retain_graph=True)

            self.q_z_optimizer.step()
            self.task_model_optimizer()
        elif mask == "bottomup":
            self.q_zxy_optimizer.zero_grad()
            GFN_loss_conditional.mean().backward(retain_graph=True)
            self.task_model_optimizer.zero_grad()
            task_model_loss = CELoss
            task_model_loss.mean().backward(retain_graph=True)

            self.p_zx_optimizer.zero_grad()
            pzx_loss = -Log_pzx
            pzx_loss.mean().backward(retain_graph=True)

            self.task_model_optimizer.step()
            self.q_zxy_optimizer.step()
            self.p_zx_optimizer.step()
        else:
            self.task_model_optimizer.zero_grad()
            task_model_loss = CELoss
            task_model_loss.mean().backward()
            self.task_model_optimizer.step()

        actual_dropout_rate = 0
        n_units = 0
        n_dropped = 0.0
        batch_size = x.shape[0]
        for layer_idx in range(len(self.mask_generator_input_shapes)):
            m = actual_masks[layer_idx]
            n_units += m.shape[1]
            n_dropped += (m == 0).float().mean(0).sum()
        actual_dropout_rate = n_dropped / n_units

        metric["actual_dropout_rate"] = actual_dropout_rate.item()
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
