
"""
 Parts of this code are based on https://github.com/tim-learn/SHOT-plus
"""

import argparse
import os
import random
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

import networks as nets
import wandb
from Trainer_adapt import TrainerG
from utils import str2bool

warnings.simplefilter("ignore", UserWarning)
username = os.environ.get("USERNAME")
torch.hub.set_dir(f'/Vols/vol_design/tools/swat/users/{username}/torch_hub_cache')

itt_delete = []


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def main():
    print('\n Setting up data sources ...')

    settings = {}
    args = parse_arguments()

    update_config(args, settings)
    torch.cuda.device(settings['gpu'])

    SEED = args.seed
    if SEED != -1:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    exp_name = settings['exp_name']

    # Test if experiment already appears in wandb.

    settings['weights_path'] = os.path.join(settings['server_root_path'], 'weights')
    print(settings['weights_path'])
    os.makedirs(os.path.join(settings['weights_path'], exp_name), exist_ok=True)

    network = build_network(settings)

    netR, optimizer = setup_optimizers(network, settings)

    update_config_wandb(args, settings)

    # ====== CALL TRAINING AND VALIDATION PROCESS ======
    trainval(network, netR, optimizer, exp_name, settings)

    wandb.finish()


def build_network(settings):
    # ====== CREATE NETWORK ======
    print('\n (build_network) Building network ...')
    num_annotators = 1
    network = nets.modnetSHOT(settings['num_C'], cnn=settings['cnn_to_use'],
                              E_dims=settings['E_dims'], apply_wn=settings['apply_wn'],
                              type_bottleneck=settings['type_bottleneck'], add_DCPL=settings['add_DCPL'],
                              normalize=not settings['additional_softmax'], num_annotators=num_annotators,
                              is_data_dependent=settings['CM_data_dependent'],
                              applySoftmax=settings['applySoftmax_CM'], beta=settings['CM_beta'],
                              use_sigmoid=settings['CM_sigmoid']).cuda()

    try:
        if settings['load_downloaded_weights']:
            # dset_name_to_path = {'OfficeHome': ,'VisDA-C':}
            dset_name = settings['dataset_name']
            if dset_name == 'OfficeHome':
                source_path = os.path.join(settings['load_downloaded_weights_path'],
                                           f"seed{settings['seed']}",
                                           'office-home',
                                           settings['source'][0].upper())
            else:
                source_path = os.path.join(settings['load_downloaded_weights_path'],
                                           f"seed{settings['seed']}",
                                           'VISDA-C/T')
            modelpath = os.path.join(source_path, 'source_F.pt')
            network.M.load_state_dict(torch.load(modelpath))
            modelpath = os.path.join(source_path, 'source_B.pt')
            network.E.load_state_dict(torch.load(modelpath))
            modelpath = os.path.join(source_path, 'source_C.pt')
            network.G.load_state_dict(torch.load(modelpath))
            print('load_downloaded_weights from ' + settings['load_downloaded_weights_path'])
        else:
            source_path = os.path.join(settings['server_root_path'], 'weights', 'DN', settings['load_exp_name'])
            modelpath = os.path.join(source_path, 'source_F.pt')
            network.M.load_state_dict(torch.load(modelpath))
            modelpath = os.path.join(source_path, 'source_B.pt')
            network.E.load_state_dict(torch.load(modelpath))
            modelpath = os.path.join(source_path, 'source_C.pt')
            network.G.load_state_dict(torch.load(modelpath))
    except:
        print('Aborting ; failed to load source model. '
              'Kindly see README for download instructions (preferred option) or train the source models.')
        exit(0)

    return network


def setup_optimizers(network, settings):
    # ====== DEFINE OPTIMIZERS ======
    print('\n Setting up optimizers ...')
    to_train = []
    if not settings['adapt_only_BN']:
        for comp in settings['optimizer']:
            if settings['to_train'][comp]:
                to_train.append(
                    {'params': network.components[comp].parameters(), 'lr': settings['lr'][comp]})
    else:
        for k, v in network.components['M'].named_parameters():
            if "bn" in k:
                to_train += [{'params': v, 'lr': settings['lr']['M']}]
            else:
                v.requires_grad = False

        for k, v in network.components['E'].named_parameters():
            to_train += [{'params': v, 'lr': settings['lr']['E']}]

        for k, v in network.components['G'].named_parameters():
            v.requires_grad = False

        if settings['add_DCPL']:
            for k, v in network.components['CM'].named_parameters():
                to_train += [{'params': v, 'lr': settings['lr']['CM']}]
    netR = []
    if settings['use_rot']:
        netR = nets.feat_classifier(type='linear', class_num=4, bottleneck_dim=2 * settings['E_dims']).cuda()
        for k, v in netR.named_parameters():
            to_train.append({'params': v, 'lr': settings['lr_value']})
    optimizer = optim.SGD(params=to_train)
    optimizer = op_copy(optimizer)
    return netR, optimizer


def update_config_wandb(args, settings):
    wandb.init(project="DCPL", name=settings["exp_name"])
    if settings["short_exp_name"] != '':
        wandb.run.summary["exp_num"] = settings["short_exp_name"]
    else:
        wandb.run.summary["exp_num"] = settings["exp_name"][-4:]
    wandb.run.summary["exp_name"] = settings["exp_name"]
    wandb.run.summary["source"] = settings["source"]
    wandb.run.summary["target"] = settings["target"]
    wandb.run.summary["dataset_name"] = settings["dataset_name"]
    wandb.run.summary["apply_wn"] = settings["apply_wn"]
    wandb.run.summary["classifier_train"] = settings['to_train']['G']
    wandb.run.summary["use_diversity_loss"] = settings["use_diversity_loss"]
    wandb.run.summary["use_entropy_loss"] = settings["use_entropy_loss"]
    wandb.run.summary["pseudo_label_classification"] = settings['use_loss']["pseudo_label_classification"]
    wandb.run.summary["seed"] = args.seed
    wandb.run.summary["interval"] = args.interval
    wandb.run.summary["PL_interval"] = args.PL_interval
    wandb.run.summary["lr_value"] = args.lr_value
    wandb.run.summary["ssl"] = args.ssl
    wandb.run.summary["use_rot"] = args.use_rot
    wandb.run.summary["per_class_acc"] = args.per_class_acc
    wandb.run.summary["apply_max"] = args.apply_max

    wandb.run.summary["cnn_to_use"] = args.cnn_to_use
    wandb.run.summary["batch_size"] = args.batch_size
    wandb.run.summary["gamma"] = args.gamma
    wandb.run.summary["adapt_only_BN"] = args.adapt_only_BN
    wandb.run.summary["add_DCPL"] = args.add_DCPL
    wandb.run.summary["CM_lr_factor"] = args.CM_lr_factor
    wandb.run.summary['use_oracle'] = args.use_oracle
    wandb.run.summary['fix_cm_visda'] = args.fix_cm_visda
    wandb.run.summary['CM_ent_loss_factor'] = args.CM_ent_loss_factor
    wandb.run.summary['add_CM_ent_loss'] = args.add_CM_ent_loss
    wandb.run.summary['CM_data_dependent'] = args.CM_data_dependent
    wandb.run.summary['trace_loss_factor'] = args.trace_loss_factor
    wandb.run.summary['CM_prior_loss'] = args.CM_prior_loss
    wandb.run.summary['CM_estimate_temp'] = args.CM_estimate_temp

    wandb.run.summary['add_trace_loss'] = args.add_trace_loss
    wandb.run.summary['CM_beta'] = args.CM_beta
    wandb.run.summary['PL_net_to_use'] = args.PL_net_to_use
    wandb.run.summary['CM_det_loss_factor'] = args.CM_det_loss_factor
    wandb.run.summary['add_CM_det_loss'] = args.add_CM_det_loss
    wandb.run.summary['PL_after_epoch'] = args.PL_after_epoch
    wandb.run.summary['only_co_learning'] = args.only_co_learning
    wandb.run.summary['with_co_learning'] = args.with_co_learning
    wandb.run.summary['CE_factor'] = args.CE_factor
    wandb.run.summary['co_learning_gamma'] = args.co_learning_gamma
    wandb.run.summary['PL_iteration_amount'] = args.PL_iteration_amount
    wandb.run.summary['PL_temperature'] = args.PL_temperature
    wandb.run.summary['CM_estimate'] = args.CM_estimate
    wandb.run.summary['CM_alpha'] = args.CM_alpha
    wandb.run.summary['CM_sigmoid'] = args.CM_sigmoid
    wandb.run.summary['subsample_data_factor'] = args.subsample_data_factor
    wandb.run.summary['PL_threshold'] = args.PL_threshold


def update_config(args, settings):
    # Update Config
    settings['PL_temperature'] = args.PL_temperature
    settings['PL_iteration_amount'] = args.PL_iteration_amount
    settings['co_learning_gamma'] = args.co_learning_gamma
    settings['with_co_learning'] = args.with_co_learning == 1
    settings['only_co_learning'] = args.only_co_learning == 1
    settings['PL_after_epoch'] = args.PL_after_epoch == 1
    settings['print_mat_for_debug'] = args.print_mat_for_debug == 1
    settings['add_CM_det_loss'] = args.add_CM_det_loss == 1
    settings['CM_det_loss_factor'] = args.CM_det_loss_factor
    settings['CM_prior_loss'] = args.CM_prior_loss
    settings['CM_beta'] = args.CM_beta
    settings['CM_sigmoid'] = args.CM_sigmoid
    settings['fix_cm_visda'] = args.fix_cm_visda
    settings['applySoftmax_CM'] = args.applySoftmax_CM == 1
    settings['CM_estimate_temp'] = args.CM_estimate_temp
    settings['add_CM_ent_loss'] = args.add_CM_ent_loss == 1
    settings['add_CM_ent_loss'] = args.add_CM_ent_loss == 1
    settings['CM_data_dependent'] = args.CM_data_dependent == 1
    settings['use_oracle'] = args.use_oracle == 1
    settings['CM_ent_loss_factor'] = args.CM_ent_loss_factor
    settings['CM_lr_factor'] = args.CM_lr_factor
    settings['PL_threshold'] = args.PL_threshold
    settings['PL_distance'] = args.PL_distance
    settings['obtain_label_by_centroids'] = args.obtain_label_by_centroids == 1
    settings['additional_softmax'] = args.additional_softmax == 1
    settings['trace_loss_factor'] = args.trace_loss_factor
    settings['diversity_loss_factor'] = args.diversity_loss_factor
    settings['add_DCPL'] = args.add_DCPL == 1
    settings['add_trace_loss'] = args.add_trace_loss == 1
    settings['PL_net_to_use'] = args.PL_net_to_use
    settings['adapt_only_BN'] = args.adapt_only_BN == 1
    settings['lr_value'] = args.lr_value
    settings['gamma'] = args.gamma
    settings['softmax_temperature'] = args.softmax_temperature
    settings['E_dims'] = args.E_dims
    settings['apply_max'] = args.apply_max == 1
    settings['img_list_root_path'] = args.img_list_root_path
    settings['ssl'] = args.ssl
    settings['start_iter'] = args.start_iter
    settings['per_class_acc'] = args.per_class_acc == 1
    settings['use_rot'] = args.use_rot == 1
    settings['short_exp_name'] = args.short_exp_name
    settings['pretrained_path'] = args.pretrained_path
    settings['data_root_path'] = args.data_root_path
    settings['server_root_path'] = args.server_root_path
    settings['cnn_to_use'] = args.cnn_to_use
    settings['interval'] = args.interval
    settings['PL_interval'] = args.PL_interval
    # settings['val_after'] = 100
    settings['load_downloaded_weights_path'] = args.load_downloaded_weights_path
    settings['load_downloaded_weights'] = args.load_downloaded_weights == 1
    settings['type_bottleneck'] = args.type_bottleneck
    settings['max_epoch'] = args.max_epoch
    settings['CE_factor'] = args.CE_factor
    # settings['num_C'] = args.num_C # Automatically deduced from dataset name.
    settings['dataset_name'] = args.dataset_name
    settings['apply_wn'] = args.apply_wn == 1
    settings['use_entropy_loss'] = args.use_entropy_loss == 1
    settings['use_diversity_loss'] = args.use_diversity_loss == 1
    # settings['dataset_exp_name'] = args.dataset_exp_name
    settings['source'] = args.source
    settings['target'] = args.target
    settings['batch_size'] = args.batch_size
    settings['CM_estimate'] = args.CM_estimate
    settings['CM_alpha'] = args.CM_alpha
    settings['num_workers'] = args.num_workers
    if args.subsample_data_factor != 1:
        print(f'WARNING!! SUBSAMPLING DATA WITH FACTOR OF {args.subsample_data_factor}')
    settings['subsample_data_factor'] = args.subsample_data_factor

    settings['optimizer_type'] = args.optimizer_type
    if settings['add_DCPL']:
        settings['optimizer'] = ['M', 'E', 'G', 'CM']
    else:
        settings['optimizer'] = ['M', 'E', 'G']
    settings['use_loss'] = {
        'adaptation': True,
        'pseudo_label_classification': args.pseudo_label_classification == 1,
    }
    if settings['add_DCPL']:
        settings['to_train'] = {
            'M': args.FE_M_train == 1,
            'E': args.E_train == 1,
            'G': args.classifier_train == 1,
            'CM': args.add_DCPL == 1
        }
    else:
        settings['to_train'] = {
            'M': args.FE_M_train == 1,
            'E': args.E_train == 1,
            'G': args.classifier_train == 1
        }
    settings['gpu'] = args.gpu
    settings['device'] = 'cuda:' + str(settings['gpu'])

    seed_to_num = {2019: 2, 2020: 1, 2021: 0}
    source = settings['source']
    target = settings['target']

    settings['seed'] = args.seed
    seed = args.seed
    dataset_name = settings['dataset_name']
    if dataset_name == 'DomainNet':
        S = source[0].upper()
        T = target[0].upper()
        load_exp_name = f'{source}_vendor_DN_mini_126_src_exp2_{seed_to_num[seed]}'
        exp_name = f'{S}to{T}_client_DCPL_DN_{seed}'  # (if DomainNet, other names for OH, Visda)
        dataset_exp_name = f'{dataset_name}/{S}to{T}_D'
        settings['num_C'] = 126
        settings['load_exp_name'] = load_exp_name
    elif dataset_name == 'OfficeHome':
        S = source[0].upper()
        T = target[0].upper()
        exp_name = f'{S}to{T}_DCPL_OH_{seed}'
        dataset_exp_name = f'{dataset_name}/{S}to{T}_D'
        settings['num_C'] = 65
    elif dataset_name == 'VisDA-C':
        dataset_exp_name = 'VisDA-C/TtoV'
        exp_name = f'DCPL_VISDA_{seed}'
        settings['num_C'] = 12
    else:
        raise Exception(f'unsupprted dataset {dataset_name}')

    settings['dataset_exp_name'] = dataset_exp_name
    settings['exp_name'] = exp_name

    # settings['exp_name'] = args.exp_name
    lr_value = args.lr_value

    settings['max_iter'] = args.max_iter
    settings['lr'] = {
        'M': lr_value * 0.1,
        'E': lr_value,
        'G': lr_value * 0.1,
        'CM': lr_value * args.CM_lr_factor,
    }
    settings['dataset_path'] = os.path.join(settings['server_root_path'], 'data', settings['dataset_exp_name'],
                                            'index_lists')
    # ======================= SANITY CHECK ======================= #
    print('######## SANITY CHECK ########')
    for key in sorted(settings.keys()):
        print('{}: {}'.format(key, settings[key]))
    # ==================== END OF SANITY CHECK ==================== #


def parse_arguments():
    parser = argparse.ArgumentParser(description='UDA')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--classifier_train', type=int, default=0)
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--FE_M_train', type=int, default=1)
    parser.add_argument('--E_train', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_diversity_loss', type=int, default=1)
    parser.add_argument('--use_entropy_loss', type=int, default=1)
    parser.add_argument('--apply_wn', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default='')
    # parser.add_argument('--num_C', type=int, default=None)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--optimizer_type', type=str, default='SGD')
    parser.add_argument('--lr_value', type=float, default=4e-3)
    parser.add_argument('--pseudo_label_classification', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--CE_factor', type=float, default=0.3)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--type_bottleneck', type=str, default="bn", help="bn or bn_relu")
    parser.add_argument('--load_downloaded_weights', type=int, default=0)
    parser.add_argument('--load_downloaded_weights_path', type=str,
                        default='weights')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--interval', type=int, default=50, help="")  # should be same as max-epoch
    parser.add_argument('--PL_interval', type=int, default=-1, help="")
    parser.add_argument('--cnn_to_use', type=str, default='resnet50', help="")
    parser.add_argument('--server_root_path', type=str,
                        default='.', help="")
    parser.add_argument('--data_root_path', type=str, default="/local_datasets/da_datasets", help="")
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--short_exp_name', type=str, default='')
    parser.add_argument('--ssl', type=float, default=0.0)
    parser.add_argument('--use_rot', type=int, default=0)
    parser.add_argument('--per_class_acc', type=int, default=0)
    parser.add_argument('--img_list_root_path', type=str, default="image_lists", help="")
    parser.add_argument('--apply_max', type=int, default=1)
    parser.add_argument('--E_dims', type=int, default=256)
    parser.add_argument('--softmax_temperature', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--adapt_only_BN', default=0, type=int)
    parser.add_argument('--PL_net_to_use', type=str, default=None, help="")
    parser.add_argument('--add_DCPL', type=int, default=1)
    parser.add_argument('--CM_estimate', type=str, default='pl_conf', choices=['pl_conf',
                                                                               'pl_conf_init', 'pl_order',
                                                                               'pl_order_init', 'pl_order_inv'])
    parser.add_argument('--CM_prior_loss', type=float, default=0.01)
    parser.add_argument('--CM_alpha', type=float, default=0.0)
    parser.add_argument('--CM_sigmoid', type=str2bool, default=False)

    parser.add_argument('--add_trace_loss', type=int, default=1)
    parser.add_argument('--diversity_loss_factor', type=float, default=1.0)
    parser.add_argument('--trace_loss_factor', type=float, default=0.01)
    parser.add_argument('--additional_softmax', type=int, default=0)
    parser.add_argument('--obtain_label_by_centroids', type=int, default=1)
    parser.add_argument('--PL_distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--PL_threshold', type=int, default=-1)
    parser.add_argument('--CM_lr_factor', type=float, default=2.0)
    parser.add_argument('--use_oracle', type=int, default=0)
    parser.add_argument('--CM_data_dependent', type=int, default=0)
    parser.add_argument('--add_CM_ent_loss', type=int, default=0)
    parser.add_argument('--CM_ent_loss_factor', type=float, default=0.0)
    parser.add_argument('--applySoftmax_CM', type=int, default=0)
    parser.add_argument('--CM_beta', type=float, default=50.0)
    parser.add_argument('--add_CM_det_loss', type=int, default=0)
    parser.add_argument('--CM_det_loss_factor', type=float, default=0.0)
    parser.add_argument('--print_mat_for_debug', type=int, default=0)
    parser.add_argument('--CM_estimate_temp', type=float, default=0.01)
    parser.add_argument('--fix_cm_visda', type=str2bool, default=False)
    parser.add_argument('--PL_after_epoch', type=int, default=0)
    parser.add_argument('--only_co_learning', type=int, default=0)
    parser.add_argument('--with_co_learning', type=int, default=0)
    parser.add_argument('--co_learning_gamma', type=float, default=0.5)
    parser.add_argument('--PL_iteration_amount', type=int, default=1)
    parser.add_argument('--PL_temperature', type=float, default=0.01)
    parser.add_argument('--subsample_data_factor', type=int, default=1, help='subsample data for debugging purposes.')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    assert args.server_root_path is not None

    if args.debug:
        args.exp_name = args.exp_name + '_debug'
        args.num_workers = 0
        args.subsample_data_factor = 50
        args.batch_size = min(args.batch_size, 64)

    if args.CM_alpha > 0 and args.CM_prior_loss > 0:
        print('CM_alpha and CM_prior_loss may not be both > 0.'
              ' Please set at least one of them to 0 and try again. Aborting run.')

    return args


def trainval(network, netR, optimizer, exp_name, settings):
    global least_val_loss
    global itt_delete

    train_iter = 0
    trainer_G = TrainerG(network, netR, optimizer, settings)
    max_iter = trainer_G.max_iter
    while True:

        trainer_G.set_mode_train()
        trainer_G.train()

        wandb.run.summary['iter_num'] = train_iter

        if train_iter % (trainer_G.interval_iter) == 0 or train_iter == max_iter:

            print("\n----------- train_iter " + str(train_iter) + ' -----------\n')
            print('validating')
            trainer_G.set_mode_val()

            if (train_iter % (trainer_G.interval_iter) == 0 or train_iter == max_iter):
                test(trainer_G, settings)

            if train_iter == max_iter:
                ############################
                # # print CM
                ############################
                # if settings['add_DCPL']:
                #     print_conf_mat(trainer_G, settings)

                dict_to_save = {component: network.components[component].cpu().state_dict() for component in
                                network.components}
                torch.save(dict_to_save, os.path.join(os.path.join(settings['weights_path'], exp_name) + '/',
                                                      'last_' + str(train_iter) + '.pth'))

            if train_iter >= max_iter:
                break

        train_iter += 1


def test(trainer_G, settings):

    if settings['dataset_name'] == 'VisDA-C':
        val_record, val_record_classes = trainer_G.val_over_val_set()
    else:
        val_record = trainer_G.val_over_val_set()

    val_acc = val_record
    trainer_G.log_errors()

    wandb.log({"val accuracy": val_acc}, step=int((trainer_G.current_iteration - 1) / trainer_G.interval_iter))

    if settings['dataset_name'] == 'VisDA-C':
        class_names = ['Plane', 'Bcycle', 'Bus', 'Car', 'Horse', 'Knife', 'Mcycl', 'Person', 'Plant',
                       'Sktbrd', 'Train', 'Truck']
        for i in range(0, len(class_names)):
            curr_str = "val acc - " + class_names[i]
            wandb.run.summary[curr_str] = val_record_classes[i]


#############################################
def print_conf_mat(trainer_G, settings):
    conf_mat = trainer_G.CM.detach().cpu().numpy() * 100

    print(conf_mat)
    plt.rcParams["figure.figsize"] = [15.0, 15.0]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()

    font_size = 20

    for i in range(trainer_G.num_C):
        for j in range(trainer_G.num_C):
            c = int(conf_mat[0, j, i])
            if c < np.max(conf_mat[0, :, :]) / 2:
                color = 'k'
            else:
                color = 'w'
            ax.text(i, j, str(c), va='center', ha='center', c=color, size=font_size)

    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])
    c = np.arange(0, 101, 10)

    ax.matshow(conf_mat[0, :, :], cmap='Blues', vmin=0, vmax=100)

    if settings['dataset_name'] == 'VisDA-C':
        label_desc = ['Plane', 'Bcycle', 'Bus', 'Car', 'Horse', 'Knife', 'Mcycl', 'Pe   rson', 'Plant',
                      'Sktbrd', 'Train', 'Truck']
    else:
        label_desc = []
        for i in range(0, settings['num_C']):
            label_desc.append(np.str(i))

    ax.set_xticklabels([''] + label_desc, fontsize=font_size)
    ax.set_yticklabels([''] + label_desc, fontsize=font_size)
    ax.locator_params(nbins=settings['num_C'], axis='x')
    ax.locator_params(nbins=settings['num_C'], axis='y')

    cbar = fig.colorbar(cmap, ticks=c)
    cbar.ax.tick_params(labelsize=font_size)

    name = "learned_conf_mat_" + str(trainer_G.current_iteration) + "_" + settings["exp_name"] + ".png"
    os.makedirs('./output/CM/', exist_ok=True)
    os.makedirs('./output/CM_NPY/', exist_ok=True)
    plt.savefig("./output/CM/" + name)
    plt.close()
    conf_mat_filename = "learned_conf_mat_" + str(trainer_G.current_iteration) + "_" + settings["exp_name"] + ".npy"

    np.save("./output/CM_NPY/" + conf_mat_filename, conf_mat)


if __name__ == '__main__':
    main()
