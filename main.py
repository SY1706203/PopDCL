import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit
from sklearn.manifold import TSNE

random.seed(101)
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
# from scipy.linalg import svd
import itertools
import torch
import time
import numpy as np
from tqdm import tqdm
from evaluator import ProxyEvaluator, GroupedEvaluator
import collections
import os
from data import Data
from parse import parse_args
from model import MF, LGN, IPS, CausE, MACR, SAMREG, INFONCE, INFONCE_batch, DCL_LOSS_batch, HCL_LOSS_batch, \
    PopDCL_LOSS_batch, BC_LOSS, BC_LOSS_batch, SimpleX, SimpleX_batch
from torch.utils.data import Dataset, DataLoader
import pickle

# import torch
# torch.cuda.empty_cache()

torch.backends.cuda.matmul.allow_tf32 = False


def merge_user_list(user_lists):
    out = collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key] = out[key] + item
    return out


def merge_user_list_no_dup(user_lists):
    out = collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key] = out[key] + item

    for key in out.keys():
        out[key] = list(set(out[key]))
    return out


def save_checkpoint(model, epoch, checkpoint_dir, buffer, max_to_keep=10):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)
    buffer.append(filename)
    if len(buffer) > max_to_keep:
        os.remove(buffer[0])
        del (buffer[0])

    return buffer


def restore_checkpoint(model, checkpoint_dir, device, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0,

    epoch_list = []

    regex = re.compile(r'\d+')

    for cp in cp_files:
        epoch_list.append([int(x) for x in regex.findall(cp)][0])

    epoch = max(epoch_list)

    if not force:
        print("Which epoch to load from? Choose in range [0, {})."
              .format(epoch), "Enter 0 to train from scratch.")
        print(">> ", end='')
        # inp_epoch = int(input())
        inp_epoch = 0
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0,
    else:
        print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(0, epoch):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename, map_location=str(device))

    try:
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
              .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch


def restore_best_checkpoint(epoch, model, checkpoint_dir, device):
    """
    Restore the best performance checkpoint
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    filename = os.path.join(checkpoint_dir,
                            'epoch={}.checkpoint.pth.tar'.format(epoch))

    print("Loading from checkpoint {}?".format(filename))

    checkpoint = torch.load(filename, map_location=str(device))

    model.load_state_dict(checkpoint['state_dict'])
    print("=> Successfully restored checkpoint (trained for {} epochs)"
          .format(checkpoint['epoch']))

    return model


def clear_checkpoint(checkpoint_dir):
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def evaluation(args, data, model, epoch, base_path, evaluator, name="valid"):
    # Evaluate with given evaluator

    ret, _ = evaluator.evaluate(model)

    n_ret = {"recall": ret[1], "hit_ratio": ret[5], "precision": ret[0], "ndcg": ret[3], "mrr": ret[4], "map": ret[2]}

    perf_str = name + ':{}'.format(n_ret)
    print(perf_str)
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(perf_str + "\n")
    # Check if need to early stop (on validation)
    is_best = False
    early_stop = False
    if name == "valid":
        if ret[1] > data.best_valid_recall:
            data.best_valid_epoch = epoch
            data.best_valid_recall = ret[1]
            data.patience = 0
            is_best = True
        else:
            data.patience += 1
            if data.patience >= args.patience:
                print_str = "The best performance epoch is % d " % data.best_valid_epoch
                print(print_str)
                early_stop = True

    return is_best, early_stop


def Item_pop(args, data, model):
    for K in range(5):
        eval_pop = ProxyEvaluator(data, data.train_user_list, data.pop_dict_list[K], top_k=[(K + 1) * 10],
                                  dump_dict=merge_user_list([data.train_user_list, data.valid_user_list]))

        ret, _ = eval_pop.evaluate(model)

        print_str = "Overlap for K = % d is % f" % ((K + 1) * 10, ret[1])

        print(print_str)

        with open('stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(print_str + "\n")


def ensureDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def split_item_grp_view(data, grp_view, grp_idx):
    n = len(grp_view)
    split_data = [{} for _ in range(n)]

    for key, item in data.items():
        for it in item:
            if key not in split_data[grp_idx[it]].keys():
                split_data[grp_idx[it]][key] = []
            split_data[grp_idx[it]][key].append(it)
    return split_data


def split_user_grp_view(data, grp_view, grp_idx):
    n = len(grp_view)
    split_data = [{} for _ in range(n)]

    for key, item in data.items():
        split_data[grp_idx[key]][key] = item

    # for key, item in data.items():
    #     for it in item:
    #         if key not in split_data[grp_idx[it]].keys():
    #             split_data[grp_idx[it]][key] = []
    #         split_data[grp_idx[it]][key].append(it)
    return split_data


def checktensor(tensor):
    t = tensor.detach().cpu().numpy()
    if np.max(np.isnan(t)):
        idx = np.argmax(np.isnan(t))
        return idx
    else:
        return -1


def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


grads = {}


def save_grad(name):
    def hook(grad):
        torch.clamp(grad, -1, 1)
        grads[name] = grad

    return hook



if __name__ == '__main__':

    start = time.time()

    args = parse_args()
    data = Data(args)
    data.load_data()
    device = "cuda:" + str(args.cuda)
    device = torch.device(args.cuda)
    saveID = args.saveID
    if args.modeltype == "INFONCE" or args.modeltype == 'INFONCE_batch':
        saveID += "n_layers=" + str(args.n_layers) + "tau=" + str(args.tau)
    if args.modeltype == "BC_LOSS" or args.modeltype == 'BC_LOSS_batch':
        saveID += "n_layers=" + str(args.n_layers) + "tau1=" + str(args.tau1) + "tau2=" + str(args.tau2) + "w=" + str(
            args.w_lambda)
    if args.modeltype == 'PopDCL_LOSS_batch':
        saveID += "tau" + str(args.Tau)
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    if args.n_layers == 2 and args.modeltype != "LGN":
        base_path = './weights/{}/{}-LGN/{}'.format(args.dataset, args.modeltype, saveID)
    else:
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    if args.modeltype == 'LGN':
        saveID += "n_layers=" + str(args.n_layers)
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    checkpoint_buffer = []
    freeze_epoch = args.freeze_epoch if (args.modeltype == "BC_LOSS" or args.modeltype == "BC_LOSS_batch") else 0
    ensureDir(base_path)

    p_item = np.array([len(data.train_item_list[u]) if u in data.train_item_list else 0 for u in range(data.n_items)])
    p_user = np.array([len(data.train_user_list[u]) if u in data.train_user_list else 0 for u in range(data.n_users)])
    m_user = np.argmax(p_user)

    np.save("pop_user", p_user)
    np.save("pop_item", p_item)

    item_pop_sorted = np.sort(p_item)
    n_groups = 3
    item_grp_view = []
    for grp in range(n_groups):
        split = int((data.n_items - 1) * (grp + 1) / n_groups)
        item_grp_view.append(item_pop_sorted[split])
    print("item_group_view:", item_grp_view)
    item_idx = np.searchsorted(item_grp_view, p_item)

    user_pop_sorted = np.sort(p_user)
    n_groups = 3
    user_grp_view = []
    for grp in range(n_groups):
        split = int((data.n_users - 1) * (grp + 1) / n_groups)
        user_grp_view.append(user_pop_sorted[split])
    print("user_group_view:", user_grp_view)
    user_idx = np.searchsorted(user_grp_view, p_user)

    eval_test_ood_item_split = split_item_grp_view(data.test_ood_user_list, item_grp_view, item_idx)
    eval_test_id_item_split = split_item_grp_view(data.test_id_user_list, item_grp_view, item_idx)
    eval_test_ood_user_split = split_user_grp_view(data.test_ood_user_list, user_grp_view, user_idx)
    eval_test_id_user_split = split_user_grp_view(data.test_id_user_list, user_grp_view, user_idx)
    # print(eval_test_id_split)
    # exit()

    item_grp_view = [0] + item_grp_view
    user_grp_view = [0] + user_grp_view

    pop_dict = {}
    for user, items in data.train_user_list.items():
        for item in items:
            if item not in pop_dict:
                pop_dict[item] = 0
            pop_dict[item] += 1

    sort_pop = sorted(pop_dict.items(), key=lambda item: item[1], reverse=True)
    pop_mask = [item[0] for item in sort_pop[:20]]
    print(pop_mask)

    if not args.pop_test:
        eval_test_ood = ProxyEvaluator(data, data.train_user_list, data.test_ood_user_list, top_k=[20],
                                       dump_dict=merge_user_list(
                                           [data.train_user_list, data.valid_user_list, data.test_id_user_list]))
        eval_test_id = ProxyEvaluator(data, data.train_user_list, data.test_id_user_list, top_k=[20],
                                      dump_dict=merge_user_list(
                                          [data.train_user_list, data.valid_user_list, data.test_ood_user_list]))
        eval_valid = ProxyEvaluator(data, data.train_user_list, data.valid_user_list, top_k=[20])

        eval_test_id_item_grouped_0 = ProxyEvaluator(data, data.train_user_list, eval_test_id_item_split[0], top_k=[20])
        eval_test_id_item_grouped_1 = ProxyEvaluator(data, data.train_user_list, eval_test_id_item_split[1], top_k=[20])
        eval_test_id_item_grouped_2 = ProxyEvaluator(data, data.train_user_list, eval_test_id_item_split[2], top_k=[20])

        eval_test_id_user_grouped_0 = ProxyEvaluator(data, data.train_user_list, eval_test_id_user_split[0], top_k=[20])
        eval_test_id_user_grouped_1 = ProxyEvaluator(data, data.train_user_list, eval_test_id_user_split[1], top_k=[20])
        eval_test_id_user_grouped_2 = ProxyEvaluator(data, data.train_user_list, eval_test_id_user_split[2], top_k=[20])

    else:
        eval_test_ood = ProxyEvaluator(data, data.train_user_list, data.test_ood_user_list, top_k=[20],
                                       dump_dict=merge_user_list(
                                           [data.train_user_list, data.valid_user_list, data.test_id_user_list]),
                                       pop_mask=pop_mask)
        eval_test_id = ProxyEvaluator(data, data.train_user_list, data.test_id_user_list, top_k=[20],
                                      dump_dict=merge_user_list(
                                          [data.train_user_list, data.valid_user_list, data.test_ood_user_list]),
                                      pop_mask=pop_mask)
        eval_valid = ProxyEvaluator(data, data.train_user_list, data.valid_user_list, top_k=[20], pop_mask=pop_mask)

    evaluators = [eval_valid, eval_test_id, eval_test_ood]
    eval_names = ["valid", "test_id", "test_ood"]
    # evaluators = [eval_valid, #eval_test_id, eval_test_ood,
    #               eval_test_id_item_grouped_0, eval_test_id_item_grouped_1, eval_test_id_item_grouped_2,
    #               eval_test_id_user_grouped_0, eval_test_id_user_grouped_1, eval_test_id_user_grouped_2
    #               ]
    # eval_names = ["valid", #"test_id", "test_ood",
    #               "test_id_item_grouped_0", "test_id_item_grouped_1", "test_id_item_grouped_2",
    #               "test_id_user_grouped_0", "test_id_user_grouped_1", "test_id_user_grouped_2"
    #               ]
    # evaluators = [eval_valid]
    # eval_names = ["valid"]

    if args.modeltype == 'LGN':
        model = LGN(args, data)
    if args.modeltype == 'INFONCE':
        model = INFONCE(args, data)
    if args.modeltype == 'INFONCE_batch':
        model = INFONCE_batch(args, data)
    if args.modeltype == 'IPS':
        model = IPS(args, data)
    if args.modeltype == 'CausE':
        model = CausE(args, data)
    if args.modeltype == 'BC_LOSS':
        model = BC_LOSS(args, data)
    if args.modeltype == 'BC_LOSS_batch':
        model = BC_LOSS_batch(args, data)
    if args.modeltype == 'MACR':
        model = MACR(args, data)
    if args.modeltype == 'SAMREG':
        model = SAMREG(args, data)
    if args.modeltype == "SimpleX":
        model = SimpleX(args, data)
    if args.modeltype == "SimpleX_batch":
        model = SimpleX_batch(args, data)

    if args.modeltype == 'PopDCL_LOSS_batch':
        model = PopDCL_LOSS_batch(args, data)

    model.cuda(device)

    model, start_epoch = restore_checkpoint(model, base_path, device)

    if args.test_only:

        for i, evaluator in enumerate(evaluators):
            is_best, temp_flag = evaluation(args, data, model, start_epoch, base_path, evaluator, eval_names[i])

        exit()

    flag = False

    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=model.lr)

    # item_pop_idx = torch.tensor(data.item_pop_idx).cuda(device)

    for epoch in range(start_epoch, args.epoch):

        # If the early stopping has been reached, restore to the best performance model
        if flag:
            break

        # All models
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        # CausE
        running_cf_loss = 0
        # BC_LOSS
        running_loss1, running_loss2 = 0, 0

        t1 = time.time()

        pbar = tqdm(enumerate(data.train_loader), total=len(data.train_loader))

        for batch_i, batch in pbar:

            batch = [x.cuda(device) for x in batch]

            users = batch[0]
            pos_items = batch[1]

            if args.modeltype != 'CausE':
                users_pop = batch[2]
                pos_items_pop = batch[3]
                pos_weights = batch[4]
                item_prior = batch[5]
                user_prior = batch[6]
                lambda_u = batch[7]
                pop_i = batch[8]
                sigma_pop_i = batch[9]
                pop_u = batch[10]
                if args.infonce == 0 or args.neg_sample != -1:
                    neg_items = batch[11]
                    neg_items_pop = batch[12]

            model.train()

            if args.modeltype == 'INFONCE_batch':

                mf_loss, reg_loss = model(users, pos_items)
                loss = mf_loss + reg_loss

            elif args.modeltype == 'INFONCE':

                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss

            elif args.modeltype == 'BC_LOSS_batch':
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = model(users, pos_items, users_pop,
                                                                               pos_items_pop)

                if epoch < args.freeze_epoch:
                    loss = loss2 + reg_loss_freeze
                else:
                    model.freeze_pop()
                    loss = loss1 + loss2 + reg_loss

            elif args.modeltype == 'BC_LOSS':
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = model(users, pos_items, neg_items, \
                                                                               users_pop, pos_items_pop, neg_items_pop)

                if epoch < args.freeze_epoch:
                    loss = loss2 + reg_loss_freeze
                else:
                    model.freeze_pop()
                    loss = loss1 + loss2 + reg_loss

            elif args.modeltype == 'IPS' or args.modeltype == 'SAMREG':

                mf_loss, reg_loss = model(users, pos_items, neg_items, pos_weights)
                loss = mf_loss + reg_loss

            elif args.modeltype == 'CausE':
                neg_items = batch[2]
                all_reg = torch.squeeze(batch[3].T.reshape([1, -1]))
                all_ctrl = torch.squeeze(batch[4].T.reshape([1, -1]))
                mf_loss, reg_loss, cf_loss = model(users, pos_items, neg_items, all_reg, all_ctrl)
                loss = mf_loss + reg_loss + cf_loss

            elif args.modeltype == "SimpleX":
                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss


            elif args.modeltype == "SimpleX_batch":
                mf_loss, reg_loss = model(users, pos_items)
                loss = mf_loss + reg_loss


            elif args.modeltype == 'PopDCL_LOSS_batch':
                mf_loss, reg_loss = model(users, pos_items, lambda_u, pop_i, sigma_pop_i)
                loss = mf_loss + reg_loss


            else:
                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()

            if args.modeltype != 'BC_LOSS' and args.modeltype != 'BC_LOSS_batch' and args.modeltype != 'Adaptive_Loss_batch':
                running_mf_loss += mf_loss.detach().item()

            num_batches += 1

        t2 = time.time()

        # Training data for one epoch

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
            epoch, t2 - t1, running_loss / num_batches,
            running_mf_loss / num_batches, running_reg_loss / num_batches)

        with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(perf_str + "\n")

        # Evaluate the trained model
        # if (epoch + 1) % 30 == 0 and epoch >= freeze_epoch:
        if (epoch + 1) % args.verbose == 0 and epoch >= freeze_epoch:
            model.eval()

            for i, evaluator in enumerate(evaluators):
                # print("now", eval_names[i])
                is_best, temp_flag = evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])

                if is_best:
                    checkpoint_buffer = save_checkpoint(model, epoch, base_path, checkpoint_buffer, args.max2keep)

                if temp_flag:
                    flag = True

        model.train()
        # visualize_embedding
        # if (epoch + 1) % 20 == 0 and epoch >= freeze_epoch:
        #     visualize_embedding(model,n=5000,epoch=epoch)

    # Get result
    model = restore_best_checkpoint(data.best_valid_epoch, model, base_path, device)
    # visualize_embedding(model,n=5000,epoch=data.best_valid_epoch)
    ensureDir("./model/{}".format(args.dataset))

    with open("./model/{}/{}.pkl".format(args.dataset, args.modeltype), 'wb') as fs:
        pickle.dump(model, fs)

    print_str = "The best epoch is % d" % data.best_valid_epoch
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")

    for i, evaluator in enumerate(evaluators[:]):
        evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")
