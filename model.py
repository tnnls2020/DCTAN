import pickle
import numpy as np
import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import load_data
import sys


class DBlock(nn.Module):
    """
    transform belief code to the μ and σ, which are the mean and variance of the Gaussian distribution.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma


class DBlock_Gcn(nn.Module):
    """
    transform belief code to the μ and σ, which are the mean and variance of the Gaussian distribution USING GCN MODEL
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock_Gcn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(input_size, hidden_size)
        self.gc_mu = GraphConvolution(hidden_size, output_size)
        self.gc_logsigma = GraphConvolution(hidden_size, output_size)

    def forward(self, x, adj):
        t = torch.tanh(self.gc1(x, adj))
        t = t * torch.sigmoid(self.gc2(x, adj))
        mu = self.gc_mu(t, adj)
        logsigma = self.gc_logsigma(t, adj)
        return mu, logsigma


class PreProcess_Gcn(nn.Module):
    """
    using gcn model to PreProcess each snapshot through graph convolution.
    fea -> feature ; hid -> hidden ; emb -> embedding ;
    """
    def __init__(self, fea_size, hid_size, emb_size, dropout):
        super(PreProcess_Gcn, self).__init__()

        self.gc1 = GraphConvolution(fea_size, hid_size)
        self.gc2 = GraphConvolution(hid_size, emb_size)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class PreProcess(nn.Module):
    """
    PreProcess the graph vector, i.e., the temporal feature vector. Do't need using the gcn.
    """

    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t


class DecoderProduct(nn.Module):
    """
    Decode the adjacency matrix and feature vector/matrix
    """

    def __init__(self, z_size, hidden_size, x_size):
        super(DecoderProduct, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z_node, z_attr):
        z_node_t = z_node.transpose(0, 1)
        adj = torch.matmul(z_node, z_node_t)
        adj = adj
        graph_feat = torch.matmul(z_attr, z_node_t)
        return adj, graph_feat


class Decoder(nn.Module):
    """
    Decode the adjacency matrix and feature matrix
    """

    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = self.fc3(t) # p = torch.sigmoid(self.fc3(t))
        return p


class MyModel(nn.Module):

    def __init__(self, node_num,
                 feat_num,
                 b_size,
                 pre_hid_size,
                 hid_size,
                 pre_out_size,
                 z_size,
                 hid_decoder_size,
                 flag,
                 delta_flag=True):
        super(MyModel, self).__init__()
        # initialize the following variables
        self.node_num = node_num
        self.attr_num = feat_num
        self.z_size = z_size
        self.b_size = b_size
        self.pre_hid_size = pre_hid_size
        self.pre_out_size = pre_out_size
        self.hid_size = hid_size
        self.hid_decoder_size = hid_decoder_size
        self.flag = flag
        self.delta_flag = delta_flag

        # preProcess the node feature by using gcn model
        self.PreProcess_gcn = PreProcess_Gcn(fea_size=self.attr_num,
                                             hid_size=self.pre_hid_size,
                                             emb_size=self.pre_out_size,
                                             dropout=0.0)

        # preProcess the feature of time by using mlp network.
        self.PreProcess = PreProcess(self.node_num,
                                     self.pre_out_size)

        # lstm for compute the belief codes for nodes.
        self.lstm_node = nn.LSTM(input_size=self.pre_out_size,
                                     hidden_size=self.b_size,
                                     batch_first=False)

        # lstm for compute the belief codes for graph vector.
        self.lstm_attr = nn.LSTM(input_size=self.pre_out_size,
                                 hidden_size=self.b_size,
                                 batch_first=False)

        # from belief code to Gaussian distribution
        self.b_to_z_node = DBlock(input_size=self.b_size,
                                  hidden_size=self.hid_size,
                                  output_size=self.z_size)
        self.b_to_z_attr = DBlock(input_size=self.b_size,
                                  hidden_size=self.hid_size,
                                  output_size=self.z_size)

        # infer the embedding at t-1 through embedding at t and belief state at t-1
        self.infer_z_node = DBlock(input_size=self.b_size + self.z_size,
                                   hidden_size=self.hid_size,
                                   output_size=self.z_size)
        self.infer_z_attr = DBlock(input_size=self.b_size + self.z_size,
                                    hidden_size=self.hid_size,
                                    output_size=self.z_size)

        # predict the future embeddings
        self.transition_node = DBlock_Gcn(input_size=self.z_size,
                                      hidden_size=self.hid_size,
                                      output_size=self.z_size)
        self.transition_attr = DBlock(input_size=self.z_size,
                                       hidden_size=self.hid_size,
                                       output_size=self.z_size)

        # reconstruction
        # self.z_to_adj = Decoder(z_size=self.z_size,
        #                         hidden_size=self.hid_decoder_size,
        #                         x_size=self.node_num)
        # 
        # self.z_to_feature = Decoder(z_size=self.z_size,
        #                             hidden_size=self.hid_decoder_size,
        #                             x_size=self.node_num)
        self.decoder_product = DecoderProduct(self.z_size, self.hid_size, x_size=self.node_num)

        self.b_node = None
        self.b_attr = None

    def forward(self, adjs, node_features, attr_features):
        self.adjs = adjs
        self.node_features = node_features # [time step, node_num, node_num]
        self.attr_features = attr_features # [time step, attr_num, node_num]

        # PreProcess
        self.preprocessed_node = self.PreProcess_gcn.forward(x=self.node_features, adj=self.adjs)
        self.preprocessed_attr = self.PreProcess.forward(input=self.attr_features)

        # lstm for obtaining belief codes for nodes and graphs, respectively.

        self.b_node, (h_n_node, c_n_node) = self.lstm_node(self.preprocessed_node)
        self.b_attr, (h_n_attr, c_n_attr) = self.lstm_attr(self.preprocessed_attr)

    def calculate_loss(self, t1, t2, adjs_ori, graph_feats_ori):

        """ sample from the learned belief based distributions at time t2"""
        # node
        z_mu_t2_belief_node, z_logsigma_t2_belief_node = self.b_to_z_node(self.b_node[t2, :, :])
        epsilon_t2_node = torch.randn_like(z_mu_t2_belief_node)
        z_t2_belief_node = z_mu_t2_belief_node + epsilon_t2_node * torch.exp(z_logsigma_t2_belief_node)

        # graph
        z_mu_t2_belief_attr, z_logsigma_t2_belief_attr = self.b_to_z_attr(self.b_attr[t2, :, :])
        epsilon_t2_attr = torch.randn_like(z_mu_t2_belief_attr)
        z_t2_belief_attr = z_mu_t2_belief_attr + epsilon_t2_attr * torch.exp(z_logsigma_t2_belief_attr)

        """ infer the latent states for nodes and graph at time step t-1 with the sampled states at t and belief state 
        at t-1"""
        # node
        z_mu_t1_infer_node, z_logsigma_t1_infer_node = self.infer_z_node(torch.cat((self.b_node[t1, :, :], z_t2_belief_node), dim=-1))
        epsilon_t1_infer_node = torch.randn_like(z_mu_t1_infer_node)
        z_t1_infer_node = z_mu_t1_infer_node + torch.exp(z_logsigma_t1_infer_node) * epsilon_t1_infer_node
        # graph
        z_mu_t1_infer_attr, z_logsigma_t1_infer_attr = self.infer_z_attr(torch.cat((self.b_attr[t1, :, :], z_t2_belief_attr), dim=-1))
        epsilon_t1_infer_attr = torch.randn_like(z_mu_t1_infer_attr)
        z_t1_infer_attr = z_mu_t1_infer_attr + torch.exp(z_logsigma_t1_infer_attr) * epsilon_t1_infer_attr

        """ belief based z distributions at time t1 """
        # node
        z_mu_t1_belief_node, z_logsigma_t1_belief_node = self.b_to_z_node(self.b_node[t1, :, :])
        # graph
        z_mu_t1_belief_attr, z_logsigma_t1_belief_attr = self.b_to_z_attr(self.b_attr[t1, :, :])

        """ predicting the future network """
        if self.delta_flag:
            # using a delta style to predict the future embeddings of nodes and attributes, i.e., mu_t2 = mu_t1 + mu_
            # delta instead of predicting directly

            # μ_t2 = μ_t1 + μ_Δ
            z_mu_t2_trans_node_delta, z_logsigma_t2_trans_node_delta = self.transition_node(z_t1_infer_node, torch.mean(self.adjs, dim=0))
            z_mu_t2_trans_attr_delta, z_logsigma_t2_trans_attr_delta = self.transition_attr(z_t1_infer_attr)
            z_mu_t2_trans_node = z_mu_t2_trans_node_delta + z_mu_t1_infer_node
            z_mu_t2_trans_attr = z_mu_t2_trans_attr_delta + z_mu_t1_infer_attr

            # σ_t2 ** 2 = σ_t1 ** 2 + σ_delta ** 2
            z_logsigma_t2_trans_node = 1/2 * torch.log((z_logsigma_t1_infer_node.exp())**2 + (z_logsigma_t2_trans_node_delta.exp())**2)
            z_logsigma_t2_trans_attr = 1/2 * torch.log((z_logsigma_t1_infer_attr.exp())**2 + (z_logsigma_t2_trans_attr_delta.exp())**2)
        else:
            z_mu_t2_trans_node, z_logsigma_t2_trans_node = self.transition_node(z_t1_infer_node,
                                                                                torch.mean(self.adjs, dim=0))
            z_mu_t2_trans_attr, z_logsigma_t2_trans_attr = self.transition_attr(z_t1_infer_attr)
        """ reconstruction """
        # adj_t2_prob = self.z_to_adj(z_t2_belief_node)
        # attribute_t2_prob = self.z_to_feature(z_t2_belief_attr)
        adj_t2_prob, attribute_t2_prob = self.decoder_product(z_t2_belief_node, z_t2_belief_attr)
        """ begin calculating loss"""
        flag_coembedding = self.flag
        adj = adjs_ori[t2]
        feature = graph_feats_ori[t2]

        # normalize parameter
        pos_weight_node = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm_node = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        pos_weight_attr = float(feature.shape[0] * feature.shape[1] - feature.sum()) / feature.sum()
        norm_attr = feature.shape[0] * feature.shape[1] / float((feature.shape[0] * feature.shape[1] - feature.sum()) * 2)
        
        criterion_node = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight_node)
        criterion_attr = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight_attr)

        """ KL-divergence"""
        alpha = 1 /self.node_num
        beta = 1 #/self.node_num

        # kl_loss += (0.5 / self.node_num * alpha) * torch.mean(torch.sum(((z_mu_t1_belief_node - z_t1_infer_node) / torch.exp(z_logsigma_t1_belief_node)) ** 2, -1) + \
        #        torch.sum(z_logsigma_t1_belief_node, -1) - torch.sum(z_logsigma_t1_infer_node, -1))
        #
        # kl_loss += (0.5 / self.feat_num * alpha) * torch.mean(torch.sum(((z_mu_t1_belief_attr - z_t1_infer_attr) / torch.exp(z_logsigma_t1_belief_attr)) ** 2, -1) + \
        #        torch.sum(z_logsigma_t1_belief_attr, -1) - torch.sum(z_logsigma_t1_infer_attr, -1))

        kl_u = (-1/2) * torch.mean(torch.sum(1 + 2 * (z_logsigma_t1_infer_node - z_logsigma_t1_belief_node)
                                             - (z_logsigma_t1_infer_node.exp() / z_logsigma_t1_belief_node.exp()) ** 2
                                             - ((z_mu_t1_belief_node - z_mu_t1_infer_node) / z_logsigma_t1_belief_node.exp())**2, -1))

        kl_a = (-1/2) * torch.mean(torch.sum(1 + 2 * (z_logsigma_t1_infer_attr - z_logsigma_t1_belief_attr)
                                             - (z_logsigma_t1_infer_attr.exp() / z_logsigma_t1_belief_attr.exp()) ** 2
                                             - ((z_mu_t1_belief_attr - z_mu_t1_infer_attr) / z_logsigma_t1_belief_attr.exp())**2, -1))
        kl_loss = kl_u * alpha + kl_a * beta

        """ log probability """
        # log state at t2 ~ z_belief_t2

        log_loss_u = alpha * torch.mean(torch.sum(-0.5 * epsilon_t2_node ** 2 - 0.5 * epsilon_t2_node.new_tensor(2 * np.pi) - z_logsigma_t2_belief_node, dim=-1))
        log_loss_a = beta * torch.mean(torch.sum(-0.5 * epsilon_t2_attr ** 2 - 0.5 * epsilon_t2_attr.new_tensor(2 * np.pi) - z_logsigma_t2_belief_attr, dim=-1))

        # log state at t1 ~ z_trans_t2
        log_loss_u += alpha * torch.mean(torch.sum(0.5 * ((z_t2_belief_node - z_mu_t2_trans_node) / torch.exp(z_logsigma_t2_trans_node)) ** 2 \
                                        + 0.5 * z_t2_belief_node.new_tensor(2 * np.pi) + z_logsigma_t2_trans_node, -1))
        log_loss_a += beta * torch.mean(torch.sum(0.5 * ((z_t2_belief_attr - z_mu_t2_trans_attr) / torch.exp(z_logsigma_t2_trans_attr)) ** 2 \
                                         + 0.5 * z_t2_belief_attr.new_tensor(2 * np.pi) + z_logsigma_t2_trans_attr, -1))
        log_loss = log_loss_a + log_loss_u
        """ reconstruction error"""
        # loss_fea_rec = torch.mean(-torch.sum(graph_feats_ori[t2, :, :] * torch.log(attribute_t2_prob) + (
        #             1 - graph_feats_ori[t2, :, :]) * torch.log(1 - attribute_t2_prob), -1))
        # loss_adj_rec = torch.mean(-torch.sum(
        #     adjs_ori[t2, :, :] * torch.log(adj_t2_prob) + (1 - adjs_ori[t2, :, :]) * torch.log(1 - adj_t2_prob),
        #     -1))

        loss_adj_rec = norm_node * criterion_node(input=adj_t2_prob, target=adjs_ori[t2, :, :])
        loss_fea_rec = norm_attr * criterion_attr(input=attribute_t2_prob, target=graph_feats_ori[t2, :, :].permute(1, 0))

        loss = (kl_loss + log_loss + loss_adj_rec + loss_fea_rec).mean()

        if not flag_coembedding:
            loss = (kl_u + log_loss_u + loss_adj_rec).mean()

        return loss, loss_fea_rec, loss_adj_rec, kl_loss, log_loss, adj_t2_prob.data.numpy(), attribute_t2_prob.data.numpy()

    def predict(self, t_final, pre_len):
        adjs_pre = []
        features_pre = []

        # sample the latent z of node for the last time step for predicting the following observations
        z_mu_belief_node, z_logsigma_belief_node = self.b_to_z_node(self.b_node[-1, :, :])
        epsilon_node = torch.randn_like(z_mu_belief_node)
        z_belief_node = z_mu_belief_node + epsilon_node * torch.exp(z_logsigma_belief_node)

        # sample the latent z of graph for the last time step for predicting the following observations
        z_mu_belief_attr, z_logsigma_belief_attr = self.b_to_z_attr(self.b_attr[-1, :, :])
        epsilon_attr = torch.randn_like(z_mu_belief_attr)
        z_belief_attr = z_mu_belief_attr + epsilon_attr * torch.exp(z_logsigma_belief_attr)

        z_node_current = z_belief_node
        z_attr_current = z_belief_attr
        z_mu_node_current,  z_logsigma_node_current = z_mu_belief_node, z_logsigma_belief_node
        z_mu_attr_current,  z_logsigma_attr_current = z_mu_belief_attr, z_logsigma_belief_attr

        adj_last, fea_last = self.decoder_product(z_node_current, z_attr_current)

        for i in range(pre_len):
            if self.delta_flag:
                print("using delta style to predict")
                z_mu_node_next_delta, z_logsigma_node_next_delta = self.transition_node(z_node_current,
                                                                            torch.mean(self.adjs, dim=0))
                z_mu_attr_next_delta, z_logsigma_attr_next_delta = self.transition_attr(z_attr_current)

                z_mu_node_next = z_mu_node_current + z_mu_node_next_delta
                z_mu_attr_next = z_mu_attr_current + z_mu_attr_next_delta
                z_logsigma_node_next = 1 / 2 * torch.log(
                    (z_logsigma_node_current.exp()) ** 2 + (z_logsigma_node_next_delta.exp()) ** 2)
                z_logsigma_attr_next = 1 / 2 * torch.log(
                    (z_logsigma_attr_current.exp()) ** 2 + (z_logsigma_attr_next_delta.exp()) ** 2)
            else:
                z_mu_node_next, z_logsigma_node_next = self.transition_node(z_node_current, torch.mean(self.adjs, dim=0))
                z_mu_attr_next, z_logsigma_attr_next = self.transition_attr(z_attr_current)

            epsilon_node = torch.randn_like(z_mu_node_next)
            epsilon_attr = torch.randn_like(z_mu_attr_next)

            """ reparameterize """
            z_node_next = z_mu_node_next + epsilon_node * torch.exp(z_logsigma_node_next)
            z_attr_next = z_mu_attr_next + epsilon_attr * torch.exp(z_logsigma_attr_next)

            """ reconstruct the future observations with the transition networks """
            adj_next, attr_next = self.decoder_product(z_node_next, z_attr_next)
            adjs_pre.append(adj_next.data.numpy())
            features_pre.append(attr_next.data.numpy())

            """ update the parameters in iteration """
            z_mu_node_current, z_logsigma_node_current = z_mu_node_next, z_logsigma_node_next
            z_mu_attr_current, z_logsigma_attr_current = z_mu_attr_next, z_logsigma_attr_next
            z_node_current = z_node_next
            z_attr_current = z_attr_next

        return np.array(adjs_pre), np.array(features_pre), adj_last.data.numpy(), fea_last.data.numpy()
