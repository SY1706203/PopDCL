from cmath import cos
import torch
import torch.nn as nn
import math
import numpy as np
from reckit import randint_choice
import random
import scipy
import torch.nn.functional as F
from tqdm import tqdm
from sparselinear import SparseLinear


def ainb(a, b):
    """gets mask for elements of a in b"""

    size = (b.size(0), a.size(0))

    if size[0] == 0:  # Prevents error in torch.Tensor.max(dim=0)
        return torch.tensor([False] * a.size(0), dtype=torch.bool)

    a = a.expand((size[0], size[1]))
    b = b.expand((size[1], size[0])).T

    mask = a.eq(b).max(dim=0).values

    return mask


def ainb_wrapper(a, b, splits=.72):
    inds = int(len(a) ** splits)

    tmp = [ainb(a[i * inds:(i + 1) * inds], b) for i in list(range(inds))]

    return torch.cat(tmp)


def slice_torch_sparse_coo_tensor(t, slices):
    """
    params:
    -------
    t: tensor to slice
    slices: slice for each dimension

    returns:
    --------
    t[slices[0], slices[1], ..., slices[n]] with size
    """

    t = t.coalesce()
    for i in range(len(slices)):
        if type(slices[i]) is not torch.Tensor:
            slices[i] = torch.tensor(slices[i], dtype=torch.long).cuda()

    indices = t.indices()
    values = t.values()
    for dim, slice in enumerate(slices):
        invert = False
        if t.size(0) * 0.6 < len(slice):
            invert = True
            all_nodes = torch.arange(t.size(0)).cuda()
            unique, counts = torch.cat([all_nodes, slice]).unique(return_counts=True)
            slice = unique[counts == 1]
        if slice.size(0) > 400:
            mask = ainb_wrapper(indices[dim], slice)
        else:
            mask = ainb(indices[dim], slice)
        if invert:
            mask = ~mask
        indices = indices[:, mask]
        values = values[mask]

    return torch.sparse_coo_tensor(indices, values, t.size()).cuda().coalesce()


class MF(nn.Module):

    def __init__(self, args, data):
        super(MF, self).__init__()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda)
        self.saveID = args.saveID
        self.f = nn.Sigmoid()

        self.train_user_list = data.train_user_list
        self.valid_user_list = data.valid_user_list
        # = torch.tensor(data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = data.user_pop_max
        self.item_pop_max = data.item_pop_max

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.normal_(self.embed_user.weight, std=0.1)
        nn.init.normal_(self.embed_item.weight, std=0.1)

        # nn.init.xavier_uniform_(self.embed_user.weight, gain=1.0)
        # nn.init.xavier_uniform_(self.embed_item.weight, gain=1.0)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = torch.transpose(self.embed_item(torch.tensor(items).cuda(self.device)), 0, 1)
        rate_batch = self.f(torch.matmul(users, items))

        return rate_batch.cpu().detach().numpy()

class LGN(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()

        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
        return negative_mask

    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        # g_droped = self.__dropout(0.8)
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        # embs = torch.stack(embs, dim=1)

        # light_out = torch.mean(embs, dim=1)
        light_out = embs[-1]
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)
        # print(negEmb0.shape)

        # users_emb = F.normalize(users_emb, dim = 1)
        # pos_emb = F.normalize(pos_emb, dim = 1)
        # neg_emb = F.normalize(neg_emb, dim = 1)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        # CCL
        # pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        # pos_scores = F.cosine_similarity(users_emb, pos_emb, dim=1)
        # print(pos_scores.shape)
        # m = 0.05
        # neg_scores = F.cosine_similarity(neg_emb, users_emb.unsqueeze(1).repeat(1, 1000, 1), dim=-1) - m
        # print(neg_scores.shape)
        # exit()
        # neg_scores = torch.bmm(neg_emb, users_emb.unsqueeze(-1)).squeeze(-1) - m
        # neg_scores = neg_scores - m
        # neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        # print(neg_scores.shape)
        # exit()

        # ui_ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))

        # mask = self.get_negative_mask(self.batch_size).cuda()
        # neg = ui_ratings.masked_select(mask).view(self.batch_size, -1)
        # pos = torch.diag(ui_ratings)

        # w=1, m=0.5
        # neg = 1 * torch.mean(torch.relu(neg_scores), dim=-1)
        # neg = torch.relu(neg_scores)

        mf_loss = torch.negative(torch.mean(maxi))
        # mf_loss = torch.mean(torch.relu(1 - pos_scores) + neg)

        # mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = self.f(torch.matmul(users, items))

        return rate_batch.cpu().detach().numpy()


class IPS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items, pos_weights):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.mul(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10), pos_weights)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class CausE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.cf_pen = args.cf_pen
        self.embed_item_ctrl = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_normal_(self.embed_item_ctrl.weight)

    def forward(self, users, pos_items, neg_items, all_reg, all_ctrl):
        all_users, all_items = self.compute()
        all_items = torch.cat([all_items, self.embed_item_ctrl.weight])

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        item_embed = all_items[all_reg]
        control_embed = all_items[all_ctrl]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        cf_loss = torch.sqrt(torch.sum(
            torch.square(torch.subtract(F.normalize(item_embed, p=2, dim=0), F.normalize(control_embed, p=2, dim=0)))))
        cf_loss = cf_loss * self.cf_pen  # / self.batch_size

        return mf_loss, reg_loss, cf_loss


class MACR(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.alpha = args.alpha
        self.beta = args.beta
        self.w = nn.Embedding(self.emb_dim, 1)
        self.w_user = nn.Embedding(self.emb_dim, 1)
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_user.weight)

        self.pos_item_scores = torch.empty((self.batch_size, 1))
        self.neg_item_scores = torch.empty((self.batch_size, 1))
        self.user_scores = torch.empty((self.batch_size, 1))

        self.rubi_c = args.c * torch.ones([1]).cuda(self.device)

    def forward(self, users, pos_items, neg_items):
        # Original scores
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        # Item module and User module
        self.pos_item_scores = torch.matmul(pos_emb, self.w.weight)
        self.neg_item_scores = torch.matmul(neg_emb, self.w.weight)
        self.user_scores = torch.matmul(users_emb, self.w_user.weight)

        # fusion
        # [batch_size,] [batch_size, 1] -> [batch_size, batch_size] * [batch_size, 1]
        # [batch_size * (bs-1)]
        pos_scores = pos_scores * torch.sigmoid(self.pos_item_scores) * torch.sigmoid(self.user_scores)
        neg_scores = neg_scores * torch.sigmoid(self.neg_item_scores) * torch.sigmoid(self.user_scores)
        # pos_scores = torch.mean(pos_scores) * torch.squeeze(torch.sigmoid(self.pos_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))
        # neg_scores = torch.mean(neg_scores) * torch.squeeze(torch.sigmoid(self.neg_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))

        # loss
        mf_loss_ori = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)))

        mf_loss_item = torch.mean(
            torch.negative(torch.log(torch.sigmoid(self.pos_item_scores) + 1e-10)) + torch.negative(
                torch.log(1 - torch.sigmoid(self.neg_item_scores) + 1e-10)))

        mf_loss_user = torch.mean(torch.negative(torch.log(torch.sigmoid(self.user_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(self.user_scores) + 1e-10)))

        mf_loss = mf_loss_ori + self.alpha * mf_loss_item + self.beta * mf_loss_user

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)

        rate_batch = torch.matmul(users, items)

        item_scores = torch.matmul(torch.transpose(items, 0, 1), self.w.weight)
        user_scores = torch.matmul(users, self.w_user.weight)

        rubi_rating_both = (rate_batch - self.rubi_c) * (torch.sigmoid(user_scores)) * torch.transpose(
            torch.sigmoid(item_scores), 0, 1)

        return rubi_rating_both.cpu().detach().numpy()


class SAMREG(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.rweight = args.rweight

    def get_correlation_loss(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm, ym = x - mx, y - my
        r_num = torch.sum(torch.mul(xm, ym))
        r_den = torch.sqrt(torch.mul(torch.sum(torch.square(xm)), torch.sum(torch.square(ym))))
        # print(r_den)
        r = r_num / (r_den + 1e-5)
        r = torch.square(torch.clamp(r, -1, 1))
        return r

    def forward(self, users, pos_items, neg_items, pop_weight):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        bpr = torch.sigmoid(pos_scores - neg_scores)

        maxi = torch.log(bpr)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        mf_loss = (1 - self.rweight) * (mf_loss + reg_loss)

        cor_loss = self.rweight * self.get_correlation_loss(pop_weight, bpr)

        return mf_loss, cor_loss


class INFONCE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim=1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss


class INFONCE_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.Tau
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1
        # add Tau+
        self.tau_plus = args.tau_plus

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            #negative_mask[i, i + batch_size] = 0
            #negative_mask[i, i + 2 * batch_size] = 0
            #negative_mask[i, i + 3 * batch_size] = 0

        # infonce++ mask
        # negative_mask = torch.cat((negative_mask, negative_mask), dim=0)

        return negative_mask

    def forward(self, users, pos_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        # origin infonce
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim=1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))


        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss


class DCL_LOSS_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Tau = args.Tau
        self.tau_plus = args.tau_plus
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
        return negative_mask

    def forward(self, users, pos_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.Tau)
        mask = self.get_negative_mask(self.batch_size).cuda()
        neg_ratings = ratings.masked_select(mask).view(self.batch_size, -1)
        neg_ratings = torch.sum(torch.exp(neg_ratings / self.Tau), dim=1)
        # denominator = torch.sum(torch.exp(ratings / self.Tau), dim=1)
        if self.tau_plus != 0:
            N = self.batch_size - 1
            Ng = (-self.tau_plus * N * numerator + neg_ratings) / (1 - self.tau_plus)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.Tau))
            dcl_loss = torch.mean(torch.negative(torch.log(numerator / (numerator + Ng))))
        else:
            dcl_loss = torch.mean(torch.negative(torch.log(numerator / (numerator + neg_ratings))))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return dcl_loss, reg_loss

class HCL_LOSS_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.hcl_beta = args.hcl_beta
        self.Tau = args.Tau
        self.tau_plus = args.tau_plus
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            # negative_mask[i, i + batch_size] = 0
            # negative_mask[i, i + 2 * batch_size] = 0
            # negative_mask[i, i + 3 * batch_size] = 0
        return negative_mask

    def forward(self, users, pos_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.Tau)
        mask = self.get_negative_mask(self.batch_size).cuda()
        neg_ratings = ratings.masked_select(mask).view(self.batch_size, -1)
        # neg_ratings = torch.sum(torch.exp(neg_ratings / self.Tau), dim=1)
        neg_ratings = torch.exp(neg_ratings / self.Tau)
        imp = (self.hcl_beta * neg_ratings.log()).exp()
        reweight_neg = (imp * neg_ratings).sum(dim=-1) / imp.mean(dim=-1)
        # denominator = torch.sum(torch.exp(ratings / self.Tau), dim=1)
        N = self.batch_size - 1
        Ng = (-self.tau_plus * N * numerator + reweight_neg) / (1 - self.tau_plus)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.Tau))
        dcl_loss = torch.mean(torch.negative(torch.log(numerator / (numerator + Ng))))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return dcl_loss, reg_loss


class PopDCL_LOSS_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Tau = args.Tau
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1


    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
        return negative_mask

    def forward(self, users, pos_items, lambda_u, pop_i, sigma_pop_i):

        mask = self.get_negative_mask(self.batch_size).cuda()

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        neg_ratings = ratings.masked_select(mask).view(self.batch_size, -1)
        pos_ratings_repeat = torch.transpose(ratings_diag.unsqueeze(0), 0, 1).repeat(1, self.batch_size - 1)

        # Compute M_plus

        # P(i not in N_u)
        P_neg = torch.div(pop_i, sigma_pop_i)

        E_neg_sum = torch.mean(neg_ratings, dim=-1) * P_neg

        M_plus=torch.sigmoid(E_neg_sum)

        ratings_diag = torch.exp((torch.diag(ratings) - M_plus) / self.Tau)

        numerator = ratings_diag

        # Compute M_minus

        # f(u,i)-f(u,j)
        diff = pos_ratings_repeat - neg_ratings

        # w+(u)/w-(u)
        lambda_u = lambda_u.unsqueeze(1).repeat(1, self.batch_size - 1)

        M_minus = lambda_u * torch.exp(diff / self.Tau)

        neg_ratings = torch.clamp(neg_ratings, min=1e-7)

        neg_beta = 1 - M_minus / neg_ratings

        # neg_beta * neg_ratings = neg_ratings - M_minus
        neg_ratings = torch.sum(torch.exp(neg_beta * neg_ratings / self.Tau), dim=1)

        Ng = neg_ratings

        PopDCL_loss = torch.mean(torch.negative(torch.log(numerator / (numerator + Ng))))

        # reg loss

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return PopDCL_loss, reg_loss


class BC_LOSS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1
        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):
        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)

        users_pop_emb = F.normalize(users_pop_emb, dim=-1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim=-1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim=-1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1),
                                   neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim=1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        pos_ratings = torch.cos(
            torch.arccos(torch.clamp(pos_ratings, -1 + 1e-7, 1 - 1e-7)) + (1 - torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim=1)

        loss1 = (1 - self.w_lambda) * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                       0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1 / self.batch_size

        regularizer2 = 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                       0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2 = regularizer2 / self.batch_size
        reg_loss = self.decay * (regularizer1 + regularizer2)

        reg_loss_freeze = self.decay * (regularizer2)
        reg_loss_norm = self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)


class BC_LOSS_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1
        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def forward(self, users, pos_items, users_pop, pos_items_pop):
        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)

        users_pop_emb = F.normalize(users_pop_emb, dim=-1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim=-1)

        ratings = torch.matmul(users_pop_emb, torch.transpose(pos_pop_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim=1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag, -1 + 1e-7, 1 - 1e-7)) + \
                                 (1 - torch.sigmoid(pos_ratings_margin)))
        '''
        theta = torch.arccos(torch.clamp(ratings_diag, -1 + 1e-7, 1 - 1e-7))
        M = torch.arccos(torch.clamp(pos_ratings_margin, -1 + 1e-7, 1 - 1e-7))
        M_ = torch.tensor([M[i] if M[i] < math.pi - theta[i] else math.pi - theta[i] for i in range(len(M))]).cuda()
        ratings_diag = torch.cos(theta + M_)
        '''

        numerator = torch.exp(ratings_diag / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim=1)
        loss1 = (1 - self.w_lambda) * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer1 = regularizer1 / self.batch_size

        regularizer2 = 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_emb) ** 2
        regularizer2 = regularizer2 / self.batch_size
        reg_loss = self.decay * (regularizer1 + regularizer2)

        reg_loss_freeze = self.decay * (regularizer2)
        reg_loss_norm = self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)


class SimpleX(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        pos_margin_loss = 1 - pos_ratings
        neg_margin_loss = torch.mean(torch.clamp(neg_ratings - self.margin, 0, 1), dim=-1)

        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class SimpleX_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample = self.batch_size - 1

    def forward(self, users, pos_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        diag_mask = torch.ones_like(ratings_diag).cuda(self.device) - torch.eye(self.batch_size).cuda(self.device)

        pos_margin_loss = 1 - ratings_diag
        neg_margin_loss = torch.sum(torch.clamp(ratings - self.margin, 0, 1) * diag_mask, dim=-1) / self.neg_sample

        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
