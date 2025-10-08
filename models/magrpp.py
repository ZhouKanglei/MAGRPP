#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2025/08/08 16:58:51

import copy
import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from utils.misc import distributed_concat
from utils.metrics import ListNetLoss
from utils.loss import compute_js_loss, pairwise_distance_loss, AngularJSDAlignLoss
from utils.loss_dino_align import DinoJSDLoss

from torch.nn.parallel import DistributedDataParallel


class Magrpp(ContinualModel):
    # magr++: extension of magr
    NAME = "magrpp"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Magrpp, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, "cpu")
        self.buffer.empty()

        self.current_task = 0
        self.i = 0
        self.lambda1 = args.alpha
        self.lambda2 = args.beta
        self.n_tasks = args.n_tasks + 1 if args.fewshot else args.n_tasks

        self.epoch_bound = [0, 20, 35]
        # self.epoch_bound = [0, 45, 46]

        # losses
        self.graph_reg_loss = AngularJSDAlignLoss()

        # optimizer
        self.build_optimizer(phase="AC")  # first train feature extractor and regressor

    def select_examples(self, lab, num):

        if len(lab) <= num:
            return list(range(len(lab)))
        else:
            selected_idxes = []

        sample_interval = len(lab) / num
        sample_ids = [int(i * sample_interval) for i in range(num)]

        scores = lab
        if len(lab.shape) == 2:
            if lab.shape[1] == 2:
                scores = lab[:, 0]
        scores = scores.reshape(
            -1,
        )
        sorted_ids = sorted(range(len(scores)), key=lambda k: scores[k])

        for i, sorted_id in enumerate(sorted_ids):
            if sorted_id in sample_ids:
                selected_idxes.append(i)

        return selected_idxes

    def fea2buffer_ous(self, dataset):
        # statistic
        examples_per_task = self.args.buffer_size // (self.n_tasks - 1)
        self.args.logging.info(
            f"Current task {self.current_task} - select {examples_per_task} samples"
        )

        # gather all labels and globally select examples
        all_labels = []
        for i, data in enumerate(dataset.train_loader):
            _, labels, _ = data

            if labels.shape[1] == 2:
                labels_ = list(labels[:, 0].numpy())
            else:
                labels_ = list(
                    labels.reshape(
                        labels.shape[0],
                    ).numpy()
                )
            all_labels.extend(labels_)

        all_labels_tensor = torch.Tensor(all_labels).to(self.device)
        all_labels = distributed_concat(all_labels_tensor).reshape(-1, 1)

        select_num = examples_per_task
        selected_idxes = self.select_examples(all_labels, select_num)
        selected_labels = all_labels[selected_idxes]
        selected_labels = list(
            selected_labels.cpu()
            .numpy()
            .reshape(
                -1,
            )
        )

        # current task
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                # select examples in a batch
                if labels.shape[1] == 2:
                    labels_ = list(labels[:, 0].numpy())
                else:
                    labels_ = list(
                        labels.reshape(
                            labels.shape[0],
                        ).numpy()
                    )

                labels_tensor = torch.Tensor(labels_).to(self.device)
                labels_ = distributed_concat(labels_tensor)

                selected_idxes = []
                for j, label in enumerate(labels_):
                    if label in selected_labels:
                        selected_idxes.append(j)
                        selected_labels.pop(selected_labels.index(label))

                selected_num = len(selected_idxes)
                if selected_num == 0:
                    continue

                selected_idxes = torch.Tensor(selected_idxes).to(self.device).long()
                # forward
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    feats = self.module.feature_extractor(inputs)
                # out = self.module.regressor(feats)
                labels = labels.to(self.device)
                # gather feats
                feats = distributed_concat(feats)[selected_idxes]
                labels = distributed_concat(labels)[selected_idxes]
                # add data
                self.buffer.add_data(
                    examples=feats.data.cpu(),
                    logits=labels.data.cpu(),
                    task_labels=torch.ones(selected_num) * (self.current_task),
                )

        # statistic
        buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
        for ttl in buf_tl.unique():
            idx = buf_tl == ttl
            if ttl > 0:
                self.args.logging.info(
                    f"Task {int(ttl)} has {sum(idx)} samples in the buffer."
                )

    def fea2buffer(self, dataset):
        examples_per_task = self.args.buffer_size // (self.n_tasks - 1)
        self.args.logging.info(
            f"Current task {self.current_task} - {examples_per_task}"
        )

        # current task
        counter = 0
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):

                if examples_per_task - counter > 0:
                    inputs, labels, not_aug_inputs = data

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    feats = self.module.feature_extractor(inputs)
                    out = self.module.regressor(feats)

                    feats = distributed_concat(feats)
                    labels = distributed_concat(labels)

                    batch_size = out.shape[0]
                    if (examples_per_task - counter) // batch_size:
                        num_select = batch_size
                    else:
                        num_select = examples_per_task - counter

                    self.buffer.add_data(
                        examples=feats.data.cpu()[:num_select],
                        logits=labels.data.cpu()[:num_select],
                        task_labels=torch.ones(num_select) * (self.current_task),
                    )

                counter += out.shape[0]

        # statistics
        buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
        for ttl in buf_tl.unique():
            idx = buf_tl == ttl

            self.args.logging.info(
                f"Task {int(ttl)} has {sum(idx)} samples in the buffer."
            )

    def update_buffer(self):
        buf_x, buf_lab, buf_tl = self.buffer.get_data(
            self.buffer.num_seen_examples, transform=self.transform
        )

        with torch.no_grad():
            buf_x = buf_x.to(self.device)
            buf_x_tilde = self.module.projector(buf_x)

        self.buffer.empty()

        self.buffer.add_data(
            examples=buf_x_tilde.data.cpu(), logits=buf_lab, task_labels=buf_tl
        )

    def _set_requires_grad(self, module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag
        pass

    def build_optimizer(self, phase: str):
        m = self.net.module if hasattr(self.net, "module") else self.net
        params = []

        # By default, freeze all parameters
        self._set_requires_grad(m.feature_extractor, False)
        self._set_requires_grad(m.regressor, False)
        self._set_requires_grad(m.projector, False)

        m.feature_extractor.eval()
        m.regressor.eval()
        m.projector.eval()

        # learning rates
        self.args.lr = 0.01 # seven 02
        fe_lr = self.args.lr * 0.1 if self.current_task else self.args.lr 
        # fe_lr = self.args.lr 
        proj_lr = self.args.lr * 0.01 if self.current_task else self.args.lr
        # proj_lr = self.args.lr if self.current_task else self.args.lr
        reg_lr = self.args.lr 

        if "A" in phase:  # phase == "A"
            # Phase A: train feature extractor
            self._set_requires_grad(m.feature_extractor, True)
            # m.feature_extractor.train()
            # freeze BN

            params.append({"params": m.feature_extractor.parameters(), "lr": fe_lr})

        if "B" in phase:  # phase == "B"
            # Phase B: train projector to learn the shift from the previous task to the current task
            self._set_requires_grad(m.projector, True)
            m.projector.train()

            params.append({"params": m.projector.parameters(), "lr": proj_lr})

        if "C" in phase:  # phase == "C"
            # Phase C: train regressor
            self._set_requires_grad(m.regressor, True)
            m.regressor.train()

            params.append({"params": m.regressor.parameters(), "lr": reg_lr})

        self.opt = torch.optim.Adam(
            params, lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        if phase in ["AB", "ABC", "B", "C", "BC"]:
            print()
        self.args.logging.info(
            f"Task {self.current_task + 1} - Phase {phase} ------------"
        )

        # compute number of parameters
        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.args.logging.info(
            f"Number of parameters in the model: {num_params:,d} ("
            f"{sum(p.numel() for p in m.feature_extractor.parameters() if p.requires_grad):,d} + "
            f"{sum(p.numel() for p in m.regressor.parameters() if p.requires_grad):,d} + "
            f"{sum(p.numel() for p in m.projector.parameters() if p.requires_grad):,d})"
        )

        # compute number of optmizer parameters for each group
        # Note: this is not the total number of parameters in the model
        # but the number of parameters that are being optimized by the optimizer
        self.args.logging.info(
            f"Optimizer parameters: {len(self.opt.param_groups)} groups"
        )
        for i, group in enumerate(self.opt.param_groups):
            num_group_params = sum(p.numel() for p in group["params"])
            self.args.logging.info(
                f"Group {i + 1} has {num_group_params:>10,d} parameters "
                f"with learning rate {group['lr']:.6f}"
            )

    def begin_epoch(self, epoch):
        if self.current_task == 0:  # first task
            if epoch == 0:
                self.task_phase = "AC"
                self.build_optimizer(phase="AC")
            return

        # after the first task, we have three phases: A, B, C
        epoch_bound = self.epoch_bound

        if epoch == epoch_bound[0]:
            self.task_phase = "A"
            self.build_optimizer(phase="A")
        elif epoch == epoch_bound[1]:
            self.task_phase = "B"
            self.build_optimizer(phase="B")
        elif epoch == epoch_bound[2]:
            self.task_phase = "BC"
            self.build_optimizer(phase="BC")

    def end_task(self, dataset):
        self.current_task += 1
        # update memory buffer
        if not self.buffer.is_empty():
            self.update_buffer()
        # add feats to buffer for replay
        if self.current_task < self.n_tasks:
            self.fea2buffer_ous(dataset)
        # copy the previous model for training the projector
        self.old_feature_extractor = copy.deepcopy(self.module.feature_extractor)

    def observe(self, inputs, labels, not_aug_inputs=None, epoch=True, task=True):

        self.i += 1
        self.opt.zero_grad()
        # forward pass
        if not self.buffer.is_empty():  # replay + pseudo-replay

            # pseudo-replay data
            with torch.no_grad():
                self.old_feature_extractor.eval()
                old_mid_feats, old_deep_feats = self.old_feature_extractor(
                    inputs, return_feats=True
                )
                old_feats = old_deep_feats.detach()

            # replay data
            buf_feats, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_feats, buf_labels = buf_feats.to(self.device), buf_labels.to(
                self.device
            )

            # loss
            if self.task_phase == "A":  # Phase A: train feature extractor
                mid_feats, deep_feats = self.module.feature_extractor(
                    inputs, return_feats=True
                )
                # intra-session graph reconstruction
                loss_j_reg = self.graph_reg_loss(deep_feats, labels)
                loss_pair = pairwise_distance_loss(deep_feats, labels)

                layer = "mixed_3c"  # 'mixed_4f' or 'mixed_5c'
                old_mid_feat = old_mid_feats[layer].detach()
                mid_feat = mid_feats[layer]

                # loss_mid = F.mse_loss(old_mid_feat, mid_feat)
                loss_mid = compute_js_loss(old_mid_feat, mid_feat)

                loss = loss_mid

            elif self.task_phase == "B":  # Phase B: train projector
                feats = self.module.feature_extractor(inputs)
                buf_hat = self.module.projector(buf_feats)
                old_hat = self.module.projector(old_feats)
                # feature alignment loss
                loss_p_fea = F.mse_loss(old_hat, feats)
                # inter-intra-joint graph reconstruction
                joint_feats = torch.cat([buf_hat, old_hat], dim=0)
                joint_labels = torch.cat([buf_labels, labels], dim=0)

                loss_j_reg = self.graph_reg_loss(joint_feats, joint_labels)

                loss = loss_p_fea + self.lambda1 * loss_j_reg

            elif self.task_phase == "C":  # Phase C: train regressor
                feats = self.module.feature_extractor(inputs)
                out = self.module.regressor(feats)
                loss_d_score = self.loss(out, labels)

                # regressor alignment
                buf_hat = self.module.projector(buf_feats)
                buf_out = self.module.regressor(buf_hat)
                loss_m_score = self.loss(buf_hat, buf_labels)

                loss = loss_d_score + loss_m_score

            elif (
                self.task_phase == "ABC" or self.task_phase == "BC"
            ):  # Phase ABC: train all parts
                # current task data
                feats = self.module.feature_extractor(inputs)
                out = self.module.regressor(feats)
                loss_d_score = self.loss(out, labels)
                # pseudo-replay data
                old_hat = self.module.projector(old_feats)
                loss_p_fea = F.mse_loss(old_hat, feats)
                # replay data
                buf_hat = self.module.projector(buf_feats)
                buf_out = self.module.regressor(buf_hat)
                loss_m_score = self.loss(buf_out, buf_labels)
                # inter-intra-joint graph reconstruction
                joint_feats = torch.cat([buf_hat, feats], dim=0)
                joint_labels = torch.cat([buf_labels, labels], dim=0)

                loss_j_reg = self.graph_reg_loss(
                    joint_feats, joint_labels, blocking=self.args.minibatch_size
                )

                loss = (
                    loss_d_score
                    + loss_m_score
                    + self.lambda1 * loss_p_fea
                    + self.lambda2 * loss_j_reg
                )

        else:

            out, feats = self.net(inputs, returnt="all")
            # feats = self.module.feature_extractor(inputs)
            # out = self.module.regressor(feats)

            loss_d_score = self.loss(out, labels)
            loss_d_reg = self.graph_reg_loss(out, labels)
            loss = loss_d_score + loss_d_reg

        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()
