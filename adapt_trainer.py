
"""
 This file was originaly copied from https://sites.google.com/view/inheritune (https://drive.google.com/file/d/1Kuec3wckMYXSrIXqh0ZXW5zHPUsQmSba/view?usp=sharing) and modified for this project needs.
 The license of the file is in: https://sites.google.com/view/inheritune (https://opensource.org/license/MIT)

 Parts of this code are based on https://github.com/tim-learn/SHOT-plus/code/uda/image_target.py
 The license of the file is in: https://github.com/tim-learn/SHOT-plus/blob/master/LICENSE
"""

import os
from os.path import join as osj

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import networks
import rotation
import wandb
from augmentations import SHOT_train_augment, SHOT_test_augment
from data_loader import ImageList, MultiEpochsDataLoader


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


######################################
# Trainer class
class Trainer():

    def __init__(self, network, netR, optimizer, settings):

        # Set the network and optimizer
        self.network = network
        self.to_train = settings['to_train']
        if settings['num_workers'] is None:
            self.num_workers = int(os.cpu_count() / torch.cuda.device_count())
        else:
            self.num_workers = settings['num_workers']
        print('num workers:', self.num_workers)

        # Optimizers to use
        self.optimizer = optimizer

        # Save the settings
        self.settings = settings

        # Get number of classes
        self.num_C = settings['num_C']

        # Extract commonly used settings
        self.batch_size = settings['batch_size']
        self.current_iteration = settings['start_iter']

        val_image_list = self.get_image_lists(settings)

        test_transform = SHOT_test_augment()

        val_dataset = ImageList(val_image_list, img_root_dir=self.settings['data_root_path'], transform=test_transform)
        batch = 64
        self.val_loader = MultiEpochsDataLoader(val_dataset, batch_size=batch, shuffle=False,
                                                num_workers=self.num_workers)

        self.set_mode_val()
        self.validation(saveToWandb=True)
        self.log_errors()
        self.set_mode_train()

        if self.settings['use_rot']:
            self.netR = netR
            netR_dict, acc_rot = self.train_target_rot()
            self.netR.load_state_dict(netR_dict)
            netR.train()

        self.pseudo_label_train = {}

        # Initialize data loaders
        self.get_all_dataloaders()

        self.max_iter = self.settings["max_epoch"] * len(self.train_loader)


        wandb.run.summary["max_iter"] = self.max_iter

        self.interval_iter = self.max_iter // self.settings["interval"]
        wandb.run.summary["interval_iter"] = self.interval_iter

        self.PL_interval_iter = self.max_iter // self.settings["PL_interval"]
        wandb.run.summary["PL_interval_iter"] = self.PL_interval_iter

        # Obtain pseudo-labels (once before training)
        self.set_mode_val()

        self.mem_label = []

        self.mem_label.append(
            torch.from_numpy(self.obtain_label_by_centroids(self.settings['PL_net_to_use'], self.settings[
                'print_mat_for_debug'])).cuda())

        if self.settings['CM_estimate'] is not None:
            cm_est = self.estimate_conf_mat_from_PL_conf(self.settings['CM_estimate'],
                                                         self.settings['CM_estimate_temp'])

            self.est_conf_mat = torch.tensor(cm_est).cuda().float()


        self.set_mode_train()


    ######################################
    def get_image_lists(self, settings):
        if settings["dataset_name"] == 'DomainNet':
            if settings["num_C"] == 40:
                image_list_file = settings["target"] + "_test_mini.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], settings["dataset_name"],
                                               "image_lists",
                                               image_list_file)
            elif self.settings["num_C"] == 126:
                image_list_file = self.settings["target"] + "_list.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], self.settings["dataset_name"],
                                               "image_lists_126",
                                               image_list_file)
            else:
                image_list_file = settings["target"] + "_test.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], settings["dataset_name"],
                                               "image_lists_full",
                                               image_list_file)
        else:
            image_list_file = settings["target"] + "_list.txt"
            image_list_path = os.path.join(self.settings['data_root_path'], settings["dataset_name"], "image_lists",
                                           image_list_file)

        image_list = open(image_list_path).readlines()
        if settings['subsample_data_factor'] != 1:
            image_list = image_list[::settings['subsample_data_factor']]
        val_image_list = image_list

        return val_image_list


    ######################################
    def estimate_conf_mat_from_PL_conf(self, conf_setting, conf_setting_temp=1.0):
        # Obtain CM estimate.
        mean_pls = []
        cur_mem_labels = self.mem_label[-1].cpu()
        if conf_setting.startswith('pl_conf'):
            if conf_setting_temp != 1.0:
                scores = torch.nn.functional.softmax(torch.from_numpy(self.similarity_PL) / conf_setting_temp, 1)
            else:
                scores = self.softmax_PL

        elif conf_setting.startswith('pl_order'):
            scores = torch.nn.functional.softmax(self.softmax_PL.argsort(1).argsort(1).float(), 1)
        elif conf_setting == 'pl_order_inv':
            scores = 1.0 / (self.num_C - self.softmax_PL.argsort(1).argsort(1).float())
        else:
            raise Exception('unsupported setting for confusion matrix estimation {conf_setting}')
        for L in range(self.num_C):
            # cur_class_confidences = self.softmax_PL[cur_mem_labels == L]
            cur_class_confidences = scores[cur_mem_labels == L]
            if len(cur_class_confidences) == 0:
                mean_pls.append(torch.ones(self.num_C) / self.num_C)
            else:
                mean_pls.append(cur_class_confidences.mean(0))
        cm_est = np.stack(mean_pls)
        cm_est /= cm_est.sum(1, keepdims=True)
        return cm_est


    ######################################
    def get_all_dataloaders(self):

        if self.settings["dataset_name"] == 'DomainNet':
            if self.settings["num_C"] == 40:
                image_list_file = self.settings["target"] + "_train_mini.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], self.settings["dataset_name"],
                                               "image_lists",
                                               image_list_file)
            elif self.settings["num_C"] == 126:
                image_list_file = self.settings["target"] + "_list.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], self.settings["dataset_name"],
                                               "image_lists_126",
                                               image_list_file)
            else:
                image_list_file = self.settings["target"] + "_train.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], self.settings["dataset_name"],
                                               "image_lists_full",
                                               image_list_file)
        else:
            image_list_file = self.settings["target"] + "_list.txt"
            image_list_path = os.path.join(self.settings['data_root_path'], self.settings["dataset_name"],
                                           "image_lists",
                                           image_list_file)

        image_list = open(image_list_path).readlines()
        train_image_list = image_list
        if self.settings['subsample_data_factor'] != 1:
            train_image_list = train_image_list[::self.settings['subsample_data_factor']]

        train_transform = SHOT_train_augment()
        test_transform = SHOT_test_augment()

        train_dataset = ImageList(train_image_list, img_root_dir=self.settings['data_root_path'],
                                         transform=train_transform)
        self.train_loader = MultiEpochsDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=self.num_workers)

        train_dataset_2 = ImageList(train_image_list, img_root_dir=self.settings['data_root_path'],
                                    transform=test_transform)
        self.PL_loader = MultiEpochsDataLoader(train_dataset_2, batch_size=self.batch_size, shuffle=False,
                                               num_workers=self.num_workers)

    ##############################
    def get_loss(self):

        ############################################
        # Cross-entropy with pseudo-labels
        ############################################

        # Co-Learning
        if self.settings['only_co_learning']:

            curr_use_PL = self.use_PL[self.img_idx]
            indices = [i for i in range(len(self.pred)) if curr_use_PL[i] == 1]

            loss_CE = nn.CrossEntropyLoss(reduction='mean')(self.features['G'][indices],
                                                            self.pseudo_label_train[0][indices])
            loss = loss_CE

        elif self.settings['PL_after_epoch'] and self.current_iteration <= self.interval_iter:
            # entropy way of pushing samples to the corresponding regions
            H_s = - torch.sum(self.softmax * torch.log(self.softmax), dim=-1)
            loss_over_batch = H_s
            loss = torch.mean(loss_over_batch, dim=0)

            # Diversity loss
            msoftmax = self.softmax.mean(dim=0)
            diversity_loss = self.settings['diversity_loss_factor'] * torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            loss -= diversity_loss

        else:
            # with DCPL (using transition matrix to deconfuse PLs)
            if self.settings['add_DCPL']:

                all_ann_losses = []
                for idx in range(len(self.pseudo_label_train)):

                    if self.settings['use_oracle']:
                        preds_annotator = torch.matmul(self.softmax, self.true_CM)
                    else:
                        if self.settings['CM_data_dependent']:
                            preds_annotator = torch.matmul(torch.unsqueeze(self.softmax, dim=-1), self.CM)
                        else:
                            preds_annotator = torch.matmul(self.softmax, self.CM[idx, :, :])

                    preds_clipped = torch.clamp(preds_annotator, 1e-10, 0.9999999)
                    preds_clipped = torch.squeeze(preds_clipped)

                    if self.settings['additional_softmax']:
                        curr_cross_entropy = nn.CrossEntropyLoss(reduction='mean')(preds_clipped,
                                                                                   self.pseudo_label_train[idx])
                    else:
                        curr_cross_entropy = torch.sum(
                            -F.one_hot(self.pseudo_label_train[idx], self.num_C) * torch.log(preds_clipped), dim=-1)

                    all_ann_losses.append(curr_cross_entropy)

                if self.settings['additional_softmax']:
                    all_ann_losses = torch.stack(all_ann_losses, dim=0)
                    cross_entropy = torch.mean(all_ann_losses)
                else:
                    losses_all_annotators = torch.stack(all_ann_losses, dim=1)
                    # cross_entropy = torch.mean(torch.sum(losses_all_annotators, dim=1))
                    cross_entropy = torch.mean(torch.mean(losses_all_annotators, dim=1))

                if self.settings['add_trace_loss']:
                    all_trace_norm = []
                    if self.settings['CM_data_dependent']:
                        for idx in range(self.CM.size()[0]):
                            all_trace_norm.append(torch.mean(torch.trace(self.CM[idx, :, :])))
                    else:
                        for idx in range(len(self.pseudo_label_train)):
                            all_trace_norm.append(torch.mean(torch.trace(self.CM[idx, :, :])))

                    all_trace_norm = torch.stack(all_trace_norm)
                    trace_norm = torch.mean(all_trace_norm)

                    # wandb.log({"cross_entropy": cross_entropy})
                    # wandb.log({"trace_norm": trace_norm})

                    loss = self.settings['CE_factor'] * cross_entropy + self.settings['trace_loss_factor'] * trace_norm


                else:
                    loss = self.settings['CE_factor'] * cross_entropy
                    # wandb.log({"cross_entropy": cross_entropy})

                if self.settings['CM_prior_loss'] > 0:
                    cm_prior_loss = torch.norm((self.CM - self.est_conf_mat).flatten()) ** 2
                    loss = loss + self.settings['CM_prior_loss'] * cm_prior_loss

            # add co-learn to SHOT
            elif self.settings['with_co_learning']:

                curr_use_PL = self.use_PL[self.img_idx]
                indices = [i for i in range(len(self.pred)) if curr_use_PL[i] == 1]
                loss_CE = nn.CrossEntropyLoss(reduction='mean')(self.features['G'][indices],
                                                                self.pseudo_label_train[0][indices])
                loss = self.settings['CE_factor'] * loss_CE


            else:
                loss_CE = nn.CrossEntropyLoss(reduction='mean')(self.features['G'],
                                                                self.pseudo_label_train[0])
                loss = self.settings['CE_factor'] * loss_CE

            #############################
            # Entropy loss
            #############################

            if self.settings['use_entropy_loss']:
                # entropy way of pushing samples to the corresponding regions
                H_s = - torch.sum(self.softmax * torch.log(self.softmax), dim=-1)
                loss_over_batch = H_s
                loss += torch.mean(loss_over_batch, dim=0)

            #############################
            # Diversity loss
            #############################
            if self.settings['use_diversity_loss']:
                msoftmax = self.softmax.mean(dim=0)
                diversity_loss = self.settings['diversity_loss_factor'] * torch.sum(
                    -msoftmax * torch.log(msoftmax + 1e-5))
                loss -= diversity_loss

            #############################
            # Rotation loss
            #############################
            if self.settings['use_rot']:
                curr_img_list = self.img_target
                r_labels_target = np.random.randint(0, 4, len(curr_img_list))
                r_inputs_target = rotation.rotate_batch_with_labels(curr_img_list, r_labels_target)
                r_labels_target = torch.from_numpy(r_labels_target).cuda()
                r_inputs_target = r_inputs_target.cuda()

                f_outputs = self.network.E(self.network.M(curr_img_list))
                f_outputs = f_outputs.detach()
                f_r_outputs = self.network.E(self.network.M(r_inputs_target))
                r_outputs_target = self.netR(torch.cat((f_outputs, f_r_outputs), 1))

                rotation_loss = self.settings['ssl'] * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
                loss += rotation_loss

        # wandb.log({"loss": loss})

        self.summary_dict['loss/'] = loss.data.cpu().numpy()

        return loss

    ####################################################
    def loss(self):

        # ==================================
        # ====== Accuracy over images ======
        # ==================================

        # Target Accuracy - all images
        concat_outputs = self.features['G']
        concat_softmax = F.softmax(concat_outputs / self.settings['softmax_temperature'], dim=-1)

        pred = torch.argmax(concat_softmax, dim=-1)

        target_acc = (pred.float() == self.gt.float()).float().mean()
        self.summary_dict['acc/target_acc'] = target_acc

        if self.phase == 'train':
            # ====== BACKPROP LOSSES ======
            self.optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            self.optimizer.step()

        self.current_iteration += 1

    ###################################################
    def forward(self):

        # Used for evaluation purposes
        self.gt = self.data[1].cuda()
        self.img_target = self.data[0][:, :3, :, :].cuda()
        self.img_idx = self.data[3]
        self.features = {}

        for idx in range(len(self.mem_label)):
            self.pseudo_label_train[idx] = self.mem_label[idx][self.img_idx]

        # Unlabeled Target data
        self.features['M'] = self.network.M(self.img_target)
        self.features['E'] = self.network.E(self.features['M'])
        self.features['G'] = self.network.G(self.features['E'])
        self.softmax = F.softmax(self.features['G'] / self.settings['softmax_temperature'], dim=-1)
        self.pred = torch.argmax(self.softmax, dim=-1)
        self.pred_conf, _ = torch.max(self.softmax, dim=-1)

        if self.settings['add_DCPL']:

            estimate_setting = self.settings['CM_estimate']
            # possibly initialize the inner matrix before calling.
            if estimate_setting is not None and self.settings['CM_prior_loss'] == 0:
                if '_init' in estimate_setting:
                    if not self.network.CM.cm_data_reinitialized:
                        self.network.CM.reinit_cm_matrix_data(self.est_conf_mat)
            self.CM = self.network.CM(self.features['E']).cuda()

            # if there's a prior loss, we do not add the matrix directly, but regularize the estimated CM to be close to it.
            if estimate_setting is not None and self.settings['CM_prior_loss'] == 0:
                if '_init' not in estimate_setting:
                    cm_alpha = self.settings['CM_alpha']
                    self.CM = (1 - cm_alpha) * self.CM + cm_alpha * self.est_conf_mat
                    # normalize again.
                    self.CM = torch.nn.functional.normalize(self.CM, dim=2, p=1)

    ##################################################
    def lr_scheduler(self, optimizer, iter_num, max_iter, gamma=10.0, power=0.75):
        decay = (1 + float(gamma * iter_num) / max_iter) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return optimizer

    ##################################################
    def train(self):

        self.phase = 'train'
        self.summary_dict = {}

        if (self.PL_interval_iter > 0 and self.current_iteration % (self.PL_interval_iter) == 0) or \
                (self.settings['PL_after_epoch'] and self.current_iteration == self.interval_iter):
            print("update PLs, iteration %d" % self.current_iteration)
            self.mem_label = []
            self.set_mode_val()
            self.mem_label.append(torch.from_numpy(self.obtain_label_by_centroids(self.settings['PL_net_to_use'],
                                                                                  self.settings[
                                                                                      'print_mat_for_debug'])).cuda())
            if self.settings['fix_cm_visda']:
                cm_est = self.estimate_conf_mat_from_PL_conf(self.settings['CM_estimate'],
                                                             self.settings['CM_estimate_temp'])
                self.est_conf_mat = torch.tensor(cm_est).cuda().float()

            self.set_mode_train()

        try:
            self.data = self.dataloader_train.__next__()[1]
            img = self.data[0]
            if img.shape[0] == 1:
                self.dataloader_train = enumerate(self.train_loader)
                self.data = self.dataloader_train.__next__()[1]
        except:
            self.dataloader_train = enumerate(self.train_loader)
            self.data = self.dataloader_train.__next__()[1]

        if self.settings['optimizer_type'] == 'SGD':
            self.lr_scheduler(self.optimizer, iter_num=self.current_iteration,
                              max_iter=self.max_iter, gamma=self.settings['gamma'])

        self.forward()
        self.loss()

        return self.summary_dict['acc/target_acc']

    ##################################################
    def log_errors(self):
        print(self.summary_dict)

    ##################################################
    def set_mode_val(self):

        self.network.eval()
        self.backward = False
        for p in self.network.parameters():
            p.requires_grad = False

    ##################################################
    def set_mode_train(self):

        self.network.train()
        self.backward = True
        for p in self.network.parameters():
            p.requires_grad = True

        if self.settings['adapt_only_BN']:
            for k, v in self.network.components['M'].named_parameters():
                if "bn" in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = False

            self.network.components['G'].eval()
            for k, v in self.network.components['G'].named_parameters():
                v.requires_grad = False

        else:
            for comp in self.settings['to_train']:
                if self.settings['to_train'][comp] == False:
                    self.network.components[comp].eval()
                    for p in self.network.components[comp].parameters():
                        p.requires_grad = False

    ####################################################
    def validation(self, saveToWandb=False):

        with torch.no_grad():
            self.summary_dict = {}

            # ----------------------
            # Target validation Data
            # ----------------------

            print('\n  target validation data')

            num_C = self.num_C

            classes = list(range(num_C))

            avg_acc = {c: 0 for c in classes}
            avg_count = {c: 0 for c in classes}

            idx = -1
            overall_acc = 0
            overall_acc_count = 0

            for data in tqdm(self.val_loader, desc='validation'):
                idx += 1
                x = data[0][:, :3, :, :].to(self.settings['device'])
                labels_target = data[1].to(self.settings['device'])

                M = self.network.components['M'](x)
                E = self.network.components['E'](M)
                G = self.network.components['G'](E)

                concat_outputs = G
                concat_softmax = F.softmax(concat_outputs / self.settings['softmax_temperature'], dim=-1)

                max_act, pred = torch.max(concat_softmax, dim=-1)

                for c in classes:
                    avg_acc[c] += (pred[labels_target == c] == labels_target[labels_target == c]).float().sum()
                    avg_count[c] += pred[labels_target == c].shape[0]

                overall_acc += (pred == labels_target).float().sum()
                overall_acc_count += pred.shape[0]

            overall_acc = float(overall_acc) / float(overall_acc_count)

            print("validation amount: ")
            print(overall_acc_count)

            # average accuracy
            avg = 0
            num_classes = num_C
            for c in classes:
                if avg_count[c] == 0:
                    avg += 0
                else:
                    avg += (float(avg_acc[c]) / float(avg_count[c]))
                    if self.settings['dataset_name'] == 'VisDA-C':
                        avg_acc[c] = (float(avg_acc[c]) / float(avg_count[c]))

            avg /= float(num_classes)
            self.summary_dict['acc/target_cs_avg'] = avg

            self.summary_dict['overall_acc/target_cs_avg'] = overall_acc

            if self.settings['dataset_name'] == 'VisDA-C':
                self.summary_dict['class_acc/target_cs_avg'] = avg_acc

            if saveToWandb:
                if self.settings['dataset_name'] == 'VisDA-C':
                    wandb.run.summary["source only acc"] = avg
                    class_names = ['Plane', 'Bcycle', 'Bus', 'Car', 'Horse', 'Knife', 'Mcycl', 'Person', 'Plant',
                                   'Sktbrd', 'Train', 'Truck']
                    for i in range(0, num_classes):
                        curr_str = "source only acc - " + class_names[i]
                        wandb.run.summary[curr_str] = avg_acc[i]

                if self.settings['dataset_name'] == 'VisDA-C' or self.settings['per_class_acc'] or self.settings[
                    'dataset_name'] == 'DomainNet':
                    wandb.run.summary["source only acc"] = avg
                    wandb.run.summary["source only overall acc"] = overall_acc

                else:
                    wandb.run.summary["source only acc"] = overall_acc

            if self.settings['dataset_name'] == 'VisDA-C':
                return avg, avg_acc

            if self.settings['per_class_acc'] or self.settings['dataset_name'] == 'DomainNet':
                wandb.run.summary["overall acc"] = overall_acc
                if self.current_iteration >= 1:
                    wandb.log({"val overall acc": overall_acc},
                              step=int((self.current_iteration - 1) / self.interval_iter))
                return avg

            return overall_acc


    ###################################################################
    def obtain_label_by_centroids(self, PL_net_to_use, print_for_debug=False):

        # ====== CREATE NETWORK ======
        print('\n (obtain_label_by_centroids) Building network ...\n')
        if PL_net_to_use != self.settings['cnn_to_use']:
            PL_network = networks.modnetSHOT(self.settings['num_C'], cnn=PL_net_to_use,
                                             E_dims=self.settings['E_dims'], apply_wn=self.settings['apply_wn'],
                                             type_bottleneck=self.settings['type_bottleneck']).to(
                self.settings['device'])

            PL_network.eval()
            for p in PL_network.parameters():
                p.requires_grad = False

        start_test = True
        with torch.no_grad():
            for data in tqdm(self.PL_loader, desc='obtain label by centroids'):
                x = data[0][:, :3, :, :].to(self.settings['device'])
                labels_target = data[1]

                # softmax output of adaptaed network
                M = self.network.components['M'](x)
                feas = self.network.components['E'](M)
                outputs = self.network.components['G'](feas)

                if PL_net_to_use != self.settings['cnn_to_use']:
                    # features of teacher network
                    feas = PL_network.components['M'](x)

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels_target.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels_target.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        predict_SR, predict = torch.max(all_output, 1)
        save_predict = predict
        save_predict_SR = predict_SR

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        if self.settings['PL_distance'] == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        ##############
        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()

        for _ in range(self.settings['PL_iteration_amount']):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count > self.settings['PL_threshold'])
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], self.settings['PL_distance'])
            if self.settings['PL_threshold'] == -1:  # we force a prediction of 0 for missing classes.
                dd[np.isnan(dd)] = 1.0
            similarity = 1 - dd
            self.similarity_PL = similarity
            self.softmax_PL = nn.Softmax(dim=1)(
                torch.from_numpy(similarity))  # Note - here we do not divide by temperature!
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
        log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

        print(log_str + '\n')

        if self.settings['with_co_learning'] or self.settings['only_co_learning']:
            self.conf_PL = torch.zeros(dd.shape[0])
            self.use_PL = torch.zeros(dd.shape[0])
            similarity = 1 - dd
            self.softmax_PL = nn.Softmax(dim=1)(torch.from_numpy(similarity / self.settings['PL_temperature']))

            for curr_img in range(dd.shape[0]):
                self.conf_PL[curr_img] = self.softmax_PL[curr_img][pred_label[curr_img]]
                if predict[curr_img] != save_predict[curr_img] and \
                        self.conf_PL[curr_img] <= self.settings['co_learning_gamma'] and save_predict_SR[curr_img] > \
                        self.settings['co_learning_gamma']:
                    predict[curr_img] = save_predict[curr_img]
                    self.use_PL[curr_img] = 1
                elif predict[curr_img] != save_predict[curr_img] \
                        and self.conf_PL[curr_img] > self.settings['co_learning_gamma'] and save_predict_SR[curr_img] <= \
                        self.settings['co_learning_gamma']:
                    self.use_PL[curr_img] = 1
                elif predict[curr_img] == save_predict[curr_img]:
                    self.use_PL[curr_img] = 1


        if self.settings['print_mat_for_debug'] or self.settings['use_oracle']:
            self.build_conf_mat(all_label, predict.astype('int'), "real", print_for_debug)


        return predict.astype('int')

    ###############################################
    def build_conf_mat(self, labels, predict, title, printMat=True):

        conf_mat_new = 100.0 * confusion_matrix(labels, predict, labels=range(self.num_C), normalize='true')
        # normalize='true' actually normalizes the rows here.
        self.true_CM = torch.tensor(conf_mat_new / 100.0,
                                    dtype=torch.float32).cuda()  # torch.from_numpy(conf_mat).cuda()

        if printMat:
            print('PRINTING CONF_MAT')
            import matplotlib as mpl

            # print(conf_mat)
            plt.rcParams["figure.figsize"] = [15.0, 15.0]
            plt.rcParams["figure.autolayout"] = True
            fig, ax = plt.subplots()

            font_size = 20

            for i in range(self.num_C):
                for j in range(self.num_C):
                    c = int(conf_mat_new[j, i])
                    if c < np.max(conf_mat_new) / 2:
                        color = 'k'
                    else:
                        color = 'w'
                    ax.text(i, j, str(c), va='center', ha='center', c=color, size=font_size)

            norm = mpl.colors.Normalize(vmin=0, vmax=100)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
            cmap.set_array([])
            c = np.arange(0, 101, 10)

            ax.matshow(conf_mat_new, cmap='Blues', vmin=0, vmax=100)

            if self.settings['dataset_name'] == 'VisDA-C':
                label_desc = ['Plane', 'Bcycle', 'Bus', 'Car', 'Horse', 'Knife', 'Mcycl', 'Person', 'Plant',
                              'Sktbrd', 'Train', 'Truck']
            else:
                label_desc = []
                for i in range(0, self.num_C):
                    label_desc.append(np.str(i))

            ax.set_xticklabels([''] + label_desc, fontsize=font_size)
            ax.set_yticklabels([''] + label_desc, fontsize=font_size)
            ax.locator_params(nbins=self.num_C, axis='x')
            ax.locator_params(nbins=self.num_C, axis='y')

            cbar = fig.colorbar(cmap, ticks=c)
            cbar.ax.tick_params(labelsize=font_size)

            name = title + "_conf_mat_" + str(self.current_iteration) + "_" + self.settings["exp_name"] + ".png"
            cm_dir = "./output/CM/"
            os.makedirs(cm_dir, exist_ok=True)
            plt.savefig(osj(cm_dir, name))
            plt.close()

            conf_mat_filename = title + "_conf_mat_" + str(self.current_iteration) + "_" + self.settings[
                "exp_name"] + ".npy"
            cm_npy_dir = "./output/CM_NPY/"
            os.makedirs(cm_npy_dir, exist_ok=True)
            np.save(osj(cm_npy_dir, conf_mat_filename), conf_mat_new)
            conf_mat_filename = title + "_conf_mat_oracle_" + str(self.current_iteration) + "_" + self.settings[
                "exp_name"] + ".pth"
            torch.save(dict(true_cm=self.true_CM.detach().cpu(),
                            labels=labels, softmax_PL=self.softmax_PL, similarity=self.similarity_PL),
                       osj(cm_npy_dir, conf_mat_filename))
            print('DONE!!')

        return conf_mat_new


    # #######################
    # # rotation classifier
    def train_target_rot(self):

        if self.settings["dataset_name"] == 'DomainNet':
            if self.settings["num_C"] == 40:
                image_list_file = self.settings["target"] + "_train_mini.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], self.settings["dataset_name"],
                                               "image_lists",
                                               image_list_file)
            elif self.settings["num_C"] == 126:
                image_list_file = self.settings["target"] + "_list.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], self.settings["dataset_name"],
                                               "image_lists_126",
                                               image_list_file)
            else:
                image_list_file = self.settings["target"] + "_train.txt"
                image_list_path = os.path.join(self.settings['img_list_root_path'], self.settings["dataset_name"],
                                               "image_lists_full",
                                               image_list_file)
        else:
            image_list_file = self.settings["target"] + "_list.txt"
            image_list_path = os.path.join(self.settings['data_root_path'], self.settings["dataset_name"],
                                           "image_lists",
                                           image_list_file)

        image_list = open(image_list_path).readlines()
        train_image_list = image_list
        train_transform = SHOT_train_augment()
        train_dataset = ImageList(train_image_list, img_root_dir=self.settings['data_root_path'],
                                  transform=train_transform)
        if self.batch_size == 256:
            curr_lr = self.settings['lr']['E'] / 4
            batch = 64
        else:
            curr_lr = self.settings['lr']['E']
            batch = self.batch_size
        train_rot_loader = MultiEpochsDataLoader(train_dataset, batch_size=batch,
                                                 shuffle=True, num_workers=self.num_workers)

        ## set base network
        if self.settings['cnn_to_use'][0:3] == 'res':
            netF = networks.ResBase(res_name=self.settings['cnn_to_use']).cuda()
        elif self.settings['cnn_to_use'] == 'SwinBase':
            netF = networks.SwinBase().cuda()

        netB = networks.feat_bootleneck(type="bn", feature_dim=netF.in_features,
                                        bottleneck_dim=self.settings['E_dims']).cuda()
        netR = networks.feat_classifier(type='linear', class_num=4, bottleneck_dim=2 * self.settings['E_dims']).cuda()

        if self.settings['load_downloaded_weights']:
            modelpath = self.settings['load_downloaded_weights_path']
        else:
            modelpath = os.path.join(self.settings['server_root_path'], 'weights', self.settings['load_exp_name'])

        netF.load_state_dict(torch.load(modelpath + '/source_F.pt'))
        netF.eval()
        for k, v in netF.named_parameters():
            v.requires_grad = False

        netB.load_state_dict(torch.load(modelpath + '/source_B.pt'))
        netB.eval()
        for k, v in netB.named_parameters():
            v.requires_grad = False

        param_group = []
        for k, v in netR.named_parameters():
            param_group += [{'params': v, 'lr': curr_lr}]
            # param_group += [{'params': v, 'lr': self.settings['lr']['E'] * 1}]
        netR.train()
        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

        max_epoch = 15
        max_iter = max_epoch * len(train_rot_loader)
        # max_iter = self.settings["max_epoch"] * len(train_rot_loader)
        interval_iter = max_iter // 10
        iter_num = 0

        rot_acc = 0
        while iter_num < max_iter:
            optimizer.zero_grad()
            try:
                data = iter_test.__next__()[1]
            except:
                iter_test = enumerate(train_rot_loader)
                data = iter_test.__next__()[1]

            inputs_test = data[0]

            if inputs_test.shape[0] == 1:
                continue

            inputs_test = inputs_test.cuda()

            iter_num += 1
            self.lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            r_labels_target = np.random.randint(0, 4, len(inputs_test))
            r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).cuda()
            r_inputs_target = r_inputs_target.cuda()

            f_outputs = netB(netF(inputs_test))
            f_r_outputs = netB(netF(r_inputs_target))
            r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
            rotation_loss.backward()

            optimizer.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netR.eval()
                acc_rot = self.cal_acc_rot(train_rot_loader, netF, netB, netR)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(self.settings['exp_name'], iter_num,
                                                                            max_iter, acc_rot)
                print(log_str + '\n')
                netR.train()

                if rot_acc < acc_rot:
                    rot_acc = acc_rot
                    best_netR = netR.state_dict()

        log_str = 'Best Accuracy = {:.2f}%'.format(rot_acc)
        print(log_str + '\n')

        del netF
        del netB
        del netR
        del train_rot_loader

        return best_netR, rot_acc

    ###################################################################
    def cal_acc_rot(self, loader, netF, netB, netR):
        start_test = True
        with torch.no_grad():
            iter_test = enumerate(loader)
            for i in range(len(loader)):
                data = iter_test.__next__()[1]
                inputs = data[0].cuda()
                r_labels = np.random.randint(0, 4, len(inputs))
                r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)
                r_labels = torch.from_numpy(r_labels)
                r_inputs = r_inputs.cuda()

                f_outputs = netB(netF(inputs))
                f_r_outputs = netB(netF(r_inputs))

                r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))
                if start_test:
                    all_output = r_outputs.float().cpu()
                    all_label = r_labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, r_labels.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        return accuracy * 100

    ###############################################
