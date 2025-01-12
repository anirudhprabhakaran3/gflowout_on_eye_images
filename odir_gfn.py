from utils.options import Options
from models import ViT_GFN
from data.ODIRDataset import ODIRImageDataset

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torchvision import models
from torchvision.utils.data import DataLoader
from torchnet import meter

import numpy as np
import scipy.stats as sts
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)

opt = Options(num_epochs=100, num_classes=8)
model = ViT_GFN(opt=opt, num_classes=8)

preprocess = models.ViT_B_16_Weights.DEFAULT.transforms(antialias=True)

training_images_path = "ODIR/training_set/Images"
training_annotations = "ODIR/training_set/Annotation/training.csv"
train_dataset = ODIRImageDataset(training_images_path, training_annotations, preprocess)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
print("Loaded training dataset")

validation_images_path = "ODIR/validation_set/Images"
validation_annotations = "ODIR/validation_set/Annotation/validation.csv"
val_dataset = ODIRImageDataset(validation_images_path, validation_annotations, preprocess)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
print("Loaded validation dataset")

def train(
        model: ViT_GFN,
        opt: Options,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader):
    model.train()

    histories = {}
    target_history = histories.get("target_history", {})
    input__history = histories.get("input__hisotry", {})
    val_accuracy_history = histories.get("val_accuracy_hisotry", {})
    first_order = histories.get("first_order_history", np.zeros(1))
    second_order = histories.get("second_order_history", np.zeros(1))
    first_order = torch.from_numpy(first_order).float().to(device)
    second_order = torch.from_numpy(second_order).float().to(device)
    variance_history = histories.get("variance_history", {})

    # def criterion(output, target_var):
    #     loss = nn.CrossEntropyLoss().to(device)(output, target_var)
    #     return loss
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.schedule_milestones, gamma=opt.lr_decay)

    loss_meter = meter.AverageValueMeter()
    accuracy_meter = meter.AverageValueMeter()
    GFNloss_unconditional_meter = meter.AverageValueMeter()
    LogZ_unconditional_meter = meter.AverageValueMeter()
    LogPF_qz_meter = meter.AverageValueMeter()
    GFNloss_conditional_meter = meter.AverageValueMeter()
    LogZ_conditional_meter = meter.AverageValueMeter()
    LogPF_qzxy_meter = meter.AverageValueMeter()
    LogPF_BNN_meter = meter.AverageValueMeter()
    actual_dropout_rate_meter = meter.AverageValueMeter()
    COR_qz_meter = meter.AverageValueMeter()
    COR_qzxy_meter = meter.AverageValueMeter()
    Log_pz_meter = meter.AverageValueMeter()
    Log_pzx_meter = meter.AverageValueMeter()

    total_steps = 0
    best_val_loss = 1e9
    epochs = opt.num_epochs

    for epoch in range(epochs):
        [print() for _ in range(5)]
        print(f"Epoch: {epoch}/{epochs}")

        model.train()
        loss_meter.reset()
        accuracy_meter.reset()

        for ii, (input_, target) in enumerate(train_dataloader):
            input_, target = input_.to(device), target.to(device)
            optimizer.zero_grad()
            
            metric = model._gfn_step(input_, target, mask=opt.mask)
            loss = metric["CELoss"]
            acc = metric["acc"]

            loss_meter.add(loss)
            accuracy_meter.add(acc)

            GFNloss_unconditional_meter.add(metric["GFN_loss_unconditional"])
            LogZ_unconditional_meter.add(metric["LogZ_unconditional"])
            LogPF_qz_meter.add(metric["LogPF_qz"])
            GFNloss_conditional_meter.add(metric["GFN_loss_conditional"])
            LogZ_conditional_meter.add(metric["LogZ_conditional"])
            LogPF_qzxy_meter.add(metric["LogPF_qzxy"])
            LogPF_BNN_meter.add(metric["LogPF_BNN"])
            actual_dropout_rate_meter.add(metric["actual_dropout_rate"])
            COR_qz_meter.add(metric["COR_qz"])
            COR_qzxy_meter.add(metric["COR_qzxy"])
            Log_pz_meter.add(metric["Log_pz"])
            Log_pzx_meter.add(metric["Log_pzx"])

            total_steps += 1

            print(
                "epoch:{epoch},lr:{lr},loss:{loss:.2f},train_acc:{train_acc:.2f} GFN_loss_conditional:{GFN_loss_conditional} GFN_loss_unconditional:{GFN_loss_unconditional} actual_dropout_rate:{actual_dropout_rate} \n".format(
                    epoch=epoch,
                    loss=loss_meter.value()[0],
                    train_acc=accuracy_meter.value()[0],
                    lr=optimizer.param_groups[0]["lr"],
                    GFN_loss_conditional=metric["GFN_loss_conditional"],
                    GFN_loss_unconditional=metric["GFN_loss_unconditional"],
                    actual_dropout_rate=metric["actual_dropout_rate"],
                )
            )


        (
            val_accuracy,
            val_loss,
            label_dict,
            input__dict,
            logits_dict,
            logits_dict_greedy,
            base_aic,
            up,
            ucpred,
            ac_prob,
            iu_prob,
            elbo,
            allMasks,
        ) = val(model, val_dataloader, criterion, opt.num_classes, opt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        val_accuracy_history[total_steps] = {
            "accuracy": val_accuracy,
            "aic": base_aic,
            "up": up,
            "ucpred": ucpred,
            "ac_prob": ac_prob,
            "iu_prob": iu_prob,
            "elbo": elbo,
        }

        scheduler.step()
        model.taskmodel_scheduler.step()

        if epoch % 20 == 0:
            model.beta = min(3.16*model.beta, 1.0) 

        print(
            "epoch:{epoch},lr:{lr},loss:{loss:.2f},val_acc:{val_acc:.2f}, uncer:{base_aic_1:.2f}, {base_aic_2:.2f},{base_aic_3:.2f}, "
            "up:{up_1:.2f}, {up_2:.2f},{up_3:.2f}, ucpred:{ucpred_1:.2f}, {ucpred_2:.2f},{ucpred_3:.2f}, "
            "ac_prob:{ac_prob_1:.2f}, {ac_prob_2:.2f},{ac_prob_3:.2f}, iu_prob:{iu_prob_1:.2f}, {iu_prob_2:.2f},{iu_prob_3:.2f}, elbo:{elbo:.2f} \n".format(
                epoch=epoch,
                loss=loss_meter.value()[0],
                val_acc=val_accuracy,
                base_aic_1=base_aic[0],
                base_aic_2=base_aic[1],
                base_aic_3=base_aic[2],
                up_1=up[0],
                up_2=up[1],
                up_3=up[2],
                ucpred_1=ucpred[0],
                ucpred_2=ucpred[1],
                ucpred_3=ucpred[2],
                ac_prob_1=ac_prob[0],
                ac_prob_2=ac_prob[1],
                ac_prob_3=ac_prob[2],
                iu_prob_1=iu_prob[0],
                iu_prob_2=iu_prob[1],
                iu_prob_3=iu_prob[2],
                elbo=elbo,
                lr=model.taskmodel_optimizer.param_groups[0]["lr"],
            )
        )
        histories["target_history"] = target_history
        histories["input__history"] = input__history
        histories["val_accuracy_history"] = val_accuracy_history
        histories["first_order_history"] = first_order.data.cpu().numpy()
        histories["second_order_history"] = second_order.data.cpu().numpy()
        histories["variance_history"] = variance_history

    print(f"Best validation loss: {best_val_loss}")


def two_sample_test_batch(logits):
    prob = torch.softmax(logits, 1)
    probmean = torch.mean(prob,2)
    values, indices = torch.topk(probmean, 2, dim=1)
    aa = logits.gather(1, indices[:,0].unsqueeze(1).unsqueeze(1).repeat(1,1,opt.sample_num))
    bb = logits.gather(1, indices[:,1].unsqueeze(1).unsqueeze(1).repeat(1,1,opt.sample_num))
    pvalue = np.zeros(shape=(aa.shape[0], aa.shape[1]))
    for i in range(pvalue.shape[0]):
        pvalue[i] = sts.wilcoxon(aa[i, 0, :], bb[i, 0, :]).pvalue
    return pvalue

def val(
        model: ViT_GFN,
        dataloader: DataLoader,
        criterion,
        num_classes,
        opt: Options):
    # also return the label (batch size), and k sampled logits (batch_size, num_classes, k)
    model.eval()
    loss_meter = meter.AverageValueMeter()
    loss_meter_greedy = meter.AverageValueMeter()
    accuracy_meter = meter.ClassErrorMeter(accuracy=True)
    accuracy_meter_greedy = meter.ClassErrorMeter(accuracy=True)
    logits_dict = OrderedDict()
    label_dict = OrderedDict()
    input__dict = OrderedDict()
    logits_dict_greedy = OrderedDict()
    accurate_pred = torch.zeros([0], dtype=torch.float64)
    testresult = torch.zeros([0], dtype=torch.float64)
    noise_mask_conca = torch.zeros([0], dtype=torch.float64)
    elbo_list = []
    label_tensors = torch.zeros([0], dtype=torch.int64)
    score_tensors = torch.zeros([0], dtype=torch.float32)

    allMasks = []
    for ii, data in enumerate(dataloader):
        input_, label = data
        input_, label = input_.to(device), label.to(device)

        logits_ii = np.zeros([input_.size(0), num_classes, opt.sample_num])
        logits_greedy = np.zeros([input_.size(0), num_classes])

        # greedy
        opt.test_sample_mode = "greedy"
        opt.use_t_in_testing = True
        noise_mask = np.zeros(shape=[input_.size(0), 1, 1, 1])

        (
            score,
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
        ) = model.gfn_forward(input_, label, mask=opt.mask)

        ####
        label_tensors = torch.cat((label_tensors, label.cpu()), 0)
        score_tensors = torch.cat((score_tensors, score.detach().cpu()), 0)
        ####
        logits_greedy[:, :] = score.cpu().data.numpy()
        logits_dict_greedy[ii] = logits_greedy
        mean_logits_greedy = torch.from_numpy(logits_greedy).to(device)
        accuracy_meter_greedy.add(mean_logits_greedy.squeeze(), label.long())
        loss_greedy = criterion(mean_logits_greedy, label)
        loss_meter_greedy.add(loss_greedy.cpu().data)
        # sample
        opt.test_sample_mode = "sample"
        opt.use_t_in_testing = False

        batch_Masks = []
        for iii in range(opt.sample_num):
            # important step !!!!!!
            outputs = model(input_, label, opt.mask)
            score, actual_masks = outputs[0], outputs[1]

            actual_masks = torch.cat(actual_masks, -1)  # shape

            batch_Masks.append(actual_masks.unsqueeze(2))

            logits_ii[:, :, iii] = score.cpu().data.numpy()
            elbo_list.append(model.elbo.cpu().numpy())

        batch_Masks = torch.cat(batch_Masks, 2)
        if ii <= 2:
            # save masks of first few batchs for later analysis
            allMasks.append(batch_Masks)

        logits_dict[ii] = logits_ii
        label_dict[ii] = label.cpu()
        input__dict[ii] = input_.cpu().numpy()
        # TODO: should I average logits or probabilities
        mean_logits = F.log_softmax(
            torch.mean(F.softmax(torch.from_numpy(logits_ii).to(device), dim=1), 2), 1
        )
        accuracy_meter.add(mean_logits.squeeze(), label.long())
        loss = criterion(mean_logits, label)
        loss_meter.add(loss.cpu().data)
        logits_tsam = torch.from_numpy(logits_ii)
        prob = F.softmax(logits_tsam, 1)
        ave_prob = torch.mean(prob, 2)
        # prediction = torch.argmax(ave_prob, 1).to(device)
        prediction = torch.argmax(torch.from_numpy(logits_greedy), 1).to(
            device
        )  # TODO: use greedy or sample?
        accurate_pred_i = (prediction == label).type_as(logits_tsam)
        accurate_pred = torch.cat([accurate_pred, accurate_pred_i], 0)
        testresult_i = torch.from_numpy(two_sample_test_batch(logits_tsam)).type_as(
            logits_tsam
        )
        testresult = torch.cat([testresult, testresult_i], 0)
        noise_mask_conca = torch.cat(
            [
                noise_mask_conca,
                torch.from_numpy(noise_mask[:, 0, 0, 0]).type_as(logits_tsam),
            ],
            0,
        )

    allMasks = torch.cat(allMasks, 2).cpu().detach().numpy()
    uncertain = (testresult > 0.01).type_as(mean_logits).cpu()
    up_1 = uncertain.mean() * 100
    ucpred_1 = ((uncertain == noise_mask_conca).type_as(mean_logits)).mean() * 100
    ac_1 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_1 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_1 = ac_1 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_1 = iu_1 / (1 - accurate_pred).sum() * 100

    uncertain = (testresult > 0.05).type_as(mean_logits).cpu()
    up_2 = uncertain.mean() * 100
    ucpred_2 = (uncertain == noise_mask_conca).type_as(mean_logits).mean() * 100
    ac_2 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_2 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_2 = ac_2 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_2 = iu_2 / (1 - accurate_pred).sum() * 100

    uncertain = (testresult > 0.1).type_as(mean_logits).cpu()
    up_3 = uncertain.mean() * 100
    ucpred_3 = (uncertain == noise_mask_conca).type_as(mean_logits).mean() * 100
    ac_3 = (accurate_pred * (1 - uncertain.squeeze())).sum()
    iu_3 = ((1 - accurate_pred) * uncertain.squeeze()).sum()

    ac_prob_3 = ac_3 / (1 - uncertain.squeeze()).sum() * 100
    iu_prob_3 = iu_3 / (1 - accurate_pred).sum() * 100

    base_aic_1 = (ac_1 + iu_1) / accurate_pred.size(0) * 100
    base_aic_2 = (ac_2 + iu_2) / accurate_pred.size(0) * 100
    base_aic_3 = (ac_3 + iu_3) / accurate_pred.size(0) * 100
    base_aic = [base_aic_1, base_aic_2, base_aic_3]

    ac_prob = [ac_prob_1, ac_prob_2, ac_prob_3]
    iu_prob = [iu_prob_1, iu_prob_2, iu_prob_3]
    ucpred = [ucpred_1, ucpred_2, ucpred_3]

    # uncertainty proportion
    up = [up_1, up_2, up_3]

    # for (i, num) in enumerate(model.get_activated_neurons() if opt.gpus <= 1 else model.module.get_activated_neurons()):
    #    vis.plot("val_layer/{}".format(i), num)

    # for (i, z_phi) in enumerate(model.z_phis()):
    #    if opt.hardsigmoid:
    #        vis.hist("hard_sigmoid(phi)/{}".format(i), F.hardtanh(opt.k * z_phi / 7. + .5, 0, 1).cpu().detach().numpy())
    #    else:
    #        vis.hist("sigmoid(phi)/{}".format(i), torch.sigmoid(opt.k * z_phi).cpu().detach().numpy())
    # if opt.gfn_dropout==False:
    #   vis.plot("prune_rate", model.prune_rate() if opt.gpus <= 1 else model.module.prune_rate())
    # return accuracy_meter.value()[0], loss_meter.value()[0], label_dict, logits_dict
    return (
        accuracy_meter_greedy.value()[0],
        loss_meter_greedy.value()[0],
        label_dict,
        input__dict,
        logits_dict,
        logits_dict_greedy,
        base_aic,
        up,
        ucpred,
        ac_prob,
        iu_prob,
        np.mean(elbo_list) * 100,
        allMasks,
    )
    # accuracy_meter.value()[0], loss_meter.value()[0]

print("Starting training....")
train(model, opt, train_dataloader, val_dataloader)