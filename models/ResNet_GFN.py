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
from utils.options import Options


class ResNetGFN(nn.Module):
    def __init__(
        self, num_classes=2, lambas=0, activation=nn.LeakyReLU, opt: Options = None
    ):
        super(ResNetGFN, self).__init__()

        self.opt = opt
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if opt.use_pretrained:
            self.resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT, num_classes=1000
            )
            self.resnet.fc = nn.Linear(512, num_classes)
        else:
            self.resnet = models.resnet18(weights=None, num_classes=num_classes)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

        # GFN Related Code
        # Adapated from Dianbo Liu
        self.random_chance = 0  # Chances of using random mask using training
        self.temperature = 2  # Temperature in sigmoid, high temperature more close the p to 0.5 for binary mask

        self.mask_generator_input_shapes = [
            (64, 32, 32),
            (64, 32, 32),
            (128, 16, 16),
            (128, 16, 16),
            (256, 8, 8),
            (256, 8, 8),
            (512, 4, 4),
            (512, 4, 4),
        ]  # Only apply dropout on the last two blocks of ResNet18

        self.rand_mask_generator = RandomMaskGenerator(dropout_rate=opt.mlp_dr)

        hiddens = [32, 32]
        self.p_zx_mask_generators = construct_conditional_mask_generators(
            n_channels=[dims[0] for dims in self.mask_generator_input_shapes],
            layer_dims=[
                dims[0] * dims[1] * dims[2] for dims in self.mask_generator_input_shapes
            ],
            additional_input_dims=[0 for j in self.mask_generator_input_shapes],
            hiddens=hiddens,
        ).to(
            self.device
        )  # p(z|x)

        self.q_zxy_mask_generators = construct_multiinput_conditional_mask_generators(
            n_channels=[dims[0] for dims in self.mask_generator_input_shapes],
            layer_dims=[
                dims[0] * dims[1] * dims[2] for dims in self.mask_generator_input_shapes
            ],
            additional_input_dims=[
                num_classes for j in self.mask_generator_input_shapes
            ],
            hiddens=hiddens,
        ).to(
            self.device
        )  # q(z | x, y)

        self.p_z_mask_generators = RandomMaskGenerator(dropout_rate=opt.mlp_dr)
        self.q_z_mask_generators = construct_unconditional_mask_generators(
            layer_dims=[x[0] for x in self.mask_generator_input_shapes]
        ).to(self.device)

        self.activation = activation

        # Paritization functions

        self.LogZ_total_flowestimator = CNN_MLP(
            CNN_in_dim=(3, self.opt.image_size, self.opt.image_size),
            mlp_in_dim=num_classes,
            out_dim=1,
            activation=nn.LeakyReLU,
        ).to(
            self.device
        )  # Paritization function when the GFN condition on both x and y

        self.LogZ_unconditional = nn.Parameter(
            torch.tensor(0.0)
        )  # Paritization funciton when GFN does not condition on any input

        # Optimizer
        z_lr = 1e-1
        mg_lr_z = 1e-3
        mg_lr_mu = 1e-3
        lr = opt.lr
        self.beta = 1  # Temperature on rewards

        q_z_param_list = [
            {
                "params": self.q_z_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            }
        ]
        self.q_z_optimizer = optim.Adam(q_z_param_list)

        p_zx_param_list = [
            {
                "params": self.p_zx_mask_generators.parameters(),
                "lr": mg_lr_mu,
                "weight_decay": 0.1,
            },
        ]
        self.p_zx_optimizer = optim.Adam(p_zx_param_list)

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
        self.q_zxy_optimizer = optim.Adam(q_zxy_param_list)

        if self.opt.tune_last_layer_only:
            task_model_param_list = [{"params": self.resnet.fc.parameters(), "lr": lr}]
        else:
            task_model_params_list = [{"params": self.resnet.parameters(), "lr": lr}]

        self.task_model_optimizer = optim.SGD(
            task_model_param_list, momentum=0.9, weight_decay=5e-4
        )
        self.task_model_scheduler = optim.lr_scheduler.MultiStepLR(
            self.task_model_optimizer, milestones=self.opt.schedule_milestone, gamma=0.1
        )

        # the following are place holder to be consistent with other codes

        self.n = 4  # 4
        self.N = 0
        self.beta_ema = 0
        self.epoch = 0
        self.elbo = torch.zeros(1)

        # droprate_init = self.opt.wrn_dr

        self.weight_decay = 0
        self.lamba = lambas

        self.to(self.device)

    def forward(self, x, y, mask="none"):
        # using GFlownet

        if (
            self.training
        ):  # be very careful it should be self.training instead of self.opt.GFN_train
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
            ####the inference code all ready set certain number of repeats, so set to 1 here
            N_repeats = 1  # sample multiple times and use average as inference prediciton because GFN cannot take expection easily
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
        ###during inference y are not used
        y = torch.nn.functional.one_hot(
            y, self.num_classes
        ).float()  # convert to one hot vector
        batch_size, input_dim = x.shape[0], x.shape[1]

        LogZ_unconditional = (
            self.LogZ_unconditional
        )  # top down mask has partition function indepdent of input x

        LogZ_conditional = self.LogZ_total_flowestimator(
            x, y
        )  # partition function for bottomp mask is input dependent

        LogPF_qz = torch.zeros(batch_size).to(
            self.device
        )  # forward probability, unconditonal mask
        LogPB_qz = torch.zeros(batch_size).to(self.device)  # backward prob
        LogR_qz = torch.zeros(batch_size).to(self.device)

        LogPF_BNN = torch.zeros(batch_size).to(self.device)
        LogPB_BNN = torch.zeros(batch_size).to(self.device)

        LogPF_qzxy = torch.zeros(batch_size).to(
            self.device
        )  # forward probability, for contional mask generator
        LogR_qzxy = torch.zeros(batch_size).to(self.device)
        LogPB_qzxy = torch.zeros(batch_size).to(self.device)

        Log_pzx = torch.zeros(batch_size).to(
            self.device
        )  # part of log R for bottom up mask
        Log_pz = torch.zeros(batch_size).to(
            self.device
        )  # part of log R for topdown mask

        """
		mu mask generation, indepedent of the input x 
		"""

        # initialize masks as all zeros(dropout them all)
        # one batch share the same mu mask

        if (
            self.training
        ):  # use tempered version of the policy q(z) or q(z|x,y) during training
            temperature = self.temperature
        else:
            temperature = 1.0

        masks_qz = [[] for _ in range(len(self.mask_generator_input_shapes))]

        for layer_idx in range(len(self.mask_generator_input_shapes)):
            if "topdown" == mask:
                EPSILON = random.uniform(0, 1)
                if layer_idx == 0:
                    # during ranom random action+ tempered policy is used
                    if (EPSILON < self.random_chance) and (self.training):
                        qz_mask_l = self.rand_mask_generator(
                            torch.zeros(
                                batch_size, self.mask_generator_input_shapes[0][0]
                            ).to(self.device)
                        ).to(self.device)

                    else:
                        qz_mask_l = self.q_z_mask_generators[layer_idx](
                            torch.zeros(batch_size, 784).to(self.device), temperature
                        )  # 784 is an arbitary number here

                    qz_p_l = self.q_z_mask_generators[layer_idx].prob(
                        torch.zeros(batch_size, 784).to(self.device), qz_mask_l
                    )
                else:
                    ##concatenate all previous masks
                    previous_mask = []
                    for j in range(layer_idx):
                        previous_mask.append(masks_qz[j][-1])
                    previous_mask = torch.cat(previous_mask, 1)

                    # during ranom random action+ tempered policy is used
                    if (EPSILON < self.random_chance) and (self.training):
                        qz_mask_l = self.rand_mask_generator(previous_mask).to(
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

                LogPF_qz += (
                    qz_mask_l * torch.log(qz_p_l)
                    + (1 - qz_mask_l) * torch.log(1 - qz_p_l)
                ).sum(1)

                LogPB_qz += 0  # uniform backward P
            else:
                masks_qz[layer_idx].append(
                    torch.ones(self.mask_generator_input_shapes[layer_idx][0]).to(
                        self.device
                    )
                )

        """
		forward pass
		"""
        actual_masks = []
        masks_conditional = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        ##resnet18 has 4 layers, each with 2 blocks
        block_idx = 0
        for layer in [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        ]:  # number of layers changes over architecture
            for blockid in range(2):  ##number block per layer changes over architecture
                identity = x
                out = layer[blockid].conv1(x)
                out = layer[blockid].bn1(out)
                out = layer[blockid].relu(out)

                out = layer[blockid].conv2(out)
                out = layer[blockid].bn2(out)

                if layer[blockid].downsample is not None:
                    identity = layer[blockid].downsample(x)

                # print("self.training",self.training)

                #####different masks generator
                if (
                    block_idx >= 0
                ):  # how many to dropout is a design choice, this version all layers dropout
                    ####if using random masks
                    EPSILON = random.uniform(0, 1)

                    # layer_idx=block_idx-6
                    layer_idx = block_idx

                    if "bottomup" in mask:
                        if self.training:
                            # during training use q(z|x,y;phi) to sample mask
                            # print("using q(z|x,y;phi)")#check the right function is used

                            if layer_idx == 0:
                                if EPSILON >= self.random_chance:
                                    m_conditional_l = self.q_zxy_mask_generators[
                                        layer_idx
                                    ](
                                        torch.zeros(batch_size, out.shape[1]).to(
                                            self.device
                                        ),
                                        out.reshape(batch_size, -1).clone().detach(),
                                        y.float().clone().detach(),
                                        temperature,
                                    )  # generate mask based on activation from previous layer, detach from BNN training
                                else:
                                    m = self.rand_mask_generator(
                                        torch.zeros(out.shape[0], out.shape[1])
                                    ).to(self.device)
                                    m_conditional_l = m

                                qzxy_p_l = self.q_zxy_mask_generators[layer_idx].prob(
                                    torch.zeros(batch_size, out.shape[1]).to(
                                        self.device
                                    ),
                                    out.reshape(batch_size, -1).clone().detach(),
                                    y.float().clone().detach(),
                                    m_conditional_l,
                                )

                            else:
                                previous_actual_mask = []  # use previous actual masks
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

                                else:  # during training ,of a centain chance a random policy will be used to explore the space
                                    m = self.rand_mask_generator(
                                        torch.zeros(out.shape[0], out.shape[1])
                                    ).to(self.device)
                                    m_conditional_l = m
                                qzxy_p_l = self.q_zxy_mask_generators[layer_idx].prob(
                                    previous_actual_mask,
                                    out.reshape(batch_size, -1).clone().detach(),
                                    y.float().clone().detach(),
                                    m_conditional_l,
                                )

                            masks_conditional.append(m_conditional_l)
                            ###add log P_F_Z to the GFN loss

                            LogPF_qzxy += (
                                m_conditional_l * torch.log(qzxy_p_l)
                                + (1 - m_conditional_l) * torch.log(1 - qzxy_p_l)
                            ).sum(1)

                            LogPB_qzxy -= 0

                        else:
                            # during inference use p(z|x;xi) to sample mask
                            # print("using p(z|x;xi)")#check the right function is used

                            if layer_idx == 0:
                                m_conditional_l = self.p_zx_mask_generators[layer_idx](
                                    out.clone().detach().reshape(out.shape[0], -1)
                                )

                            else:
                                previous_actual_mask = []  # use previous actual masks
                                for j in range(layer_idx):
                                    previous_actual_mask.append(actual_masks[j])
                                # print("layer_idx",layer_idx)
                                # print("out",out.shape)
                                # print("actual_masks[j]",actual_masks[j].shape)
                                ###calculate p(z|x;xi)
                                input_pzx = torch.cat(
                                    previous_actual_mask
                                    + [out.clone().detach().reshape(out.shape[0], -1)],
                                    1,
                                )
                                # print("input_pzx",input_pzx.shape,x.shape)
                                m_conditional_l = self.p_zx_mask_generators[layer_idx](
                                    input_pzx
                                )  # generate mask based on activation from previous layer, detach from BNN training

                            masks_conditional.append(m_conditional_l)

                    else:
                        masks_conditional.append(torch.ones(out.shape).to(self.device))

                    if mask == "random":  # completely random mask used
                        EPSILON = random.uniform(0, 1)
                        m = self.rand_mask_generator(
                            torch.zeros(out.shape[0], out.shape[1])
                        ).to(self.device)

                    elif mask == "topdown":
                        m_qz_l = masks_qz[layer_idx][-1]
                        m = m_qz_l
                    elif mask == "bottomup":
                        m = m_conditional_l

                    elif mask == "none":
                        m = torch.ones(out.shape[0], out.shape[1]).to(self.device)

                    m = (
                        m.detach().clone()
                    )  # mask should not interfere backbone model(MLP or resnet etc) training

                    ###calculate p(z;xi) and p(z|x;xi) both of which are needed for training
                    if layer_idx == 0:
                        ###calculate p(z|x;xi)
                        Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(
                            out.reshape(batch_size, -1).clone().detach(), m
                        )
                        # calculate p(z|xi)
                        Log_P_z_l = self.p_z_mask_generators.log_prob(m, m)

                    else:
                        previous_actual_mask = []  # use previous actual masks
                        for j in range(layer_idx):
                            previous_actual_mask.append(actual_masks[j])

                        ###calculate p(z|x;xi)
                        input_pzx = torch.cat(
                            previous_actual_mask
                            + [out.reshape(batch_size, -1).clone().detach()],
                            1,
                        )

                        Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(
                            input_pzx, m
                        )  # generate mask based on activation from previous layer, detach from BNN training

                        ###calculate p(z;xi)

                        Log_P_z_l = self.p_z_mask_generators.log_prob(
                            m, m
                        )  # generate mask based on activation from previous layer, detach from BNN training

                    Log_pzx += Log_P_zx_l
                    Log_pz += Log_P_z_l

                    ###apply the mask

                    out = out.mul(m.unsqueeze(2).unsqueeze(3))

                    actual_masks.append(m)

                out += identity
                out = layer[blockid].relu(out)
                x = out

                block_idx += 1

        ####fc layer

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        pred = x
        # pred = F.log_softmax(x, dim=1)

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

    def _gfn_step(self, x, y, mask_train="", mask="none"):
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

        # loss calculation
        # CEloss = F.nll_loss(logits, y)
        # CEloss = F.nll_loss(reduction='none')(logits, y)
        CEloss = nn.CrossEntropyLoss(reduction="none")(logits, y)
        LL = -CEloss

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

        # Update model

        acc = (torch.argmax(logits, dim=1) == y).sum().item() / len(y)

        metric["CELoss"] = CEloss.mean().item()  # loss output from the model

        metric[
            "GFN_loss_unconditional"
        ] = (
            GFN_loss_unconditional.mean().item()
        )  # loss output from the model is per sample
        metric[
            "GFN_loss_conditional"
        ] = (
            GFN_loss_conditional.mean().item()
        )  # loss output from the model is per sample

        metric["acc"] = acc

        COR_qz = np.corrcoef(
            LogPF_qz.cpu().detach().numpy(), LogR_unconditional.cpu().detach().numpy()
        )[0, 1]

        metric["COR_qz"] = COR_qz

        COR_qzxy = np.corrcoef(
            LogPF_qzxy.cpu().detach().numpy(), LogR_conditional.cpu().detach().numpy()
        )[0, 1]

        metric["COR_qzxy"] = COR_qzxy

        if mask == "topdown":
            # train  q(z) and logZ by GFN loss
            self.q_z_optimizer.zero_grad()

            GFN_loss_unconditional.mean().backward(retain_graph=True)

            self.task_model_optimizer.zero_grad()

            taskmodel_loss = CEloss
            taskmodel_loss.mean().backward(retain_graph=True)

            # self.task_model_optimizer.step()

            self.q_z_optimizer.step()
            self.task_model_optimizer.step()

        if mask == "bottomup":
            # train  q(z|x,y) and logZ by GFN loss
            self.q_zxy_optimizer.zero_grad()

            GFN_loss_conditional.mean().backward(retain_graph=True)
            self.task_model_optimizer.zero_grad()

            taskmodel_loss = CEloss
            taskmodel_loss.mean().backward(retain_graph=True)

            # self.task_model_optimizer.step()

            ##train p(z|x) by maximize EBLO

            self.p_zx_optimizer.zero_grad()
            pzx_loss = -Log_pzx
            pzx_loss.mean().backward(retain_graph=True)
            self.task_model_optimizer.step()
            self.q_zxy_optimizer.step()
            self.p_zx_optimizer.step()

        if mask == "random" or mask == "none":
            self.task_model_optimizer.zero_grad()

            taskmodel_loss = CEloss
            taskmodel_loss.mean().backward()

            self.task_model_optimizer.step()

        ###calculated actual droppout rate
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

        # differnet terms of TB
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
