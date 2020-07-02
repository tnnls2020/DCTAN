import torch
from model import *
from utils import *
import load_data
import utils
import numpy as np
import argparse
from utils import get_roc_score_adj, get_roc_score_feat


def train(args):
    dataset = args.dataset
    print("loading data from :{}".format(dataset))
    adjs, features = load_data.load_data(dataset)
    node_num = adjs.shape[1]
    attribute_num = features.shape[2]
    time_length = adjs.shape[0]
    print("finish loading: node_number:{} ; time_length:{}; attribute_number:{}".format(node_num, time_length, attribute_num))

    """  set parameters """
    pre_len = args.pre_len  # how many time steps to predict.
    train_len = time_length - pre_len

    """ preProcess data """
    # preserve original data
    adjs_ori = torch.from_numpy(adjs).type(torch.float) + torch.eye(node_num)
    feats_ori = torch.from_numpy(features).type(torch.float).type(torch.float)

    # process data

    # divide testing/validating/training sets
    adjs_train, val_adjs, val_adjs_negative, test_adjs, test_adjs_negative = utils.mask_adjs_test(adjs=adjs)
    fea_train, val_feas, val_feas_false, \
    test_feas, test_feas_false = utils.mask_attributes_test(features)

    adjs_train_lable = torch.from_numpy(adjs_train).type(torch.float) + torch.eye(node_num)
    adjs_train = utils.preprocess_adjs(adjs_train)
    adjs_train = torch.from_numpy(adjs_train).type(torch.float)

    # node_features = torch.eye(node_num).unsqueeze(0).repeat(time_length, 1, 1) 单位矩阵作为特征
    node_features = torch.from_numpy(features).type(torch.float)
    attributes = torch.from_numpy(features.transpose([0, 2, 1])).type(torch.float)  # batch_size = 1

    """ implement a CDN model """
    myModel = MyModel(node_num=node_num,
                      feat_num=attribute_num,
                      b_size=args.belief_size,
                      pre_hid_size=args.pre_hidden_size,
                      hid_size=args.hidden_size,
                      pre_out_size=args.pre_out_size,
                      z_size=args.emb_size,
                      hid_decoder_size=args.decoder_hidden,
                      flag=args.co_embedding)

    # Adam Optimizer
    optimizer = optim.Adam(myModel.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    # begin training
    print("="*30)
    print("begin training")
    for epoch in range(args.epochs):
        myModel.train()
        optimizer.zero_grad()
        myModel.forward(adjs=adjs_train[0:train_len],
                        node_features=node_features[0:train_len],
                        attr_features=attributes[0:train_len])
        # random choose two successive time steps t1 and t2
        t_1 = np.random.choice(train_len - 1)
        t_2 = t_1 + np.random.choice([1])
        loss, loss_fea_rec, loss_adj_rec, kl_loss, log_loss, adj_t2_prob, feature_t2_prob \
            = myModel.calculate_loss(t_1, t_2, adjs_ori=adjs_train_lable, graph_feats_ori=feats_ori)

        # test on validate test
        roc_adj, ap_adj = get_roc_score_adj(val_adjs[t_2], val_adjs_negative[t_2], adj_t2_prob, t_2, adjs_ori)
        roc_feat, ap_feat = get_roc_score_feat(val_feas[t_2], val_feas_false[t_2], feature_t2_prob, t_2, feats_ori)
        print("epoch:{} loss_train:{:.5f} "
              "t1:{:2} t2:{:2} "
              "loss_fea_rec:{:.5f} loss_adj_rec:{:.5f} "
              "kl_loss:{:.5f} log_loss:{:.5f} "
              "roc_adj:{:.5f} ap_adj:{:.5f}"
              "roc_fea:{:.5f} ap_fea:{:.5f} ".format(epoch, loss.item(),
                                                     t_1, t_2,
                                                     loss_fea_rec, loss_adj_rec,
                                                     kl_loss, log_loss,
                                                     roc_adj, ap_adj,
                                                     roc_feat, ap_feat
                                                     ))

        # update the dynamic network
        if epoch % 300 == 0:
            print("=" * 30)
            print("begin testing")
            print(" time_length : {} train_length : {} predict_length : {}".format(time_length, train_len, pre_len))

            # predict the future observations
            adjs_pre, features_pre, adj_last, features_last = myModel.predict(t_final=-1, pre_len=pre_len)

            # calculate the scores
            for t in range(train_len, time_length):
                adj_t_prob = adjs_pre[t - train_len]
                feature_t_prob = features_pre[t - train_len]
                roc_adj, ap_adj = get_roc_score_adj(val_adjs[t], val_adjs_negative[t], adj_t_prob, t, adjs_ori)
                roc_feat, ap_feat = get_roc_score_feat(val_feas[t], val_feas_false[t], feature_t_prob, t, feats_ori)
                print(" roc_adj:{:.5f} ap_adj:{:.5f}"
                      " roc_fea:{:.5f} ap_adj:{:.5f} ".format(roc_adj, ap_adj, roc_feat, ap_feat))

            # using the last embedding to reconstruct and predict the links and associations
            print("using the last time")
            for t in range(train_len-1, time_length):
                adj_t_prob = adj_last
                feature_t_prob = features_last
                roc_adj, ap_adj = get_roc_score_adj(val_adjs[t], val_adjs_negative[t], adj_t_prob, t, adjs_ori)
                roc_feat, ap_feat = get_roc_score_feat(val_feas[t], val_feas_false[t], feature_t_prob, t, feats_ori)
                print(" roc_adj:{:.5f} ap_adj:{:.5f}"
                      " roc_fea:{:.5f} ap_adj:{:.5f} ".format(roc_adj, ap_adj, roc_feat, ap_feat))
            print("finish testing")

            print("=" * 30)

        # update parameters
        loss.backward()
        optimizer.step()
    print("="*30)
    print("finish training")

    print("="*30)
    print("begin testing")
    print(" time_length : {} train_length : {} predict_length : {}".format(time_length, train_len, pre_len))

    # predict the future observations
    adjs_pre, features_pre, adj_last, features_last = myModel.predict(t_final=-1, pre_len=pre_len)

    # calculate the scores
    print("using the delta way to predict")
    for t in range(train_len, time_length):
        adj_t_prob = adjs_pre[t - train_len]
        feature_t_prob = features_pre[t - train_len]
        roc_adj, ap_adj = get_roc_score_adj(test_adjs[t], test_adjs_negative[t], adj_t_prob, t, adjs_ori)
        roc_feat, ap_feat = get_roc_score_feat(test_feas[t], test_feas_false[t], feature_t_prob, t, feats_ori)
        print(" roc_adj:{:.5f} ap_adj:{:.5f}"
              " roc_fea:{:.5f} ap_adj:{:.5f} ".format(roc_adj, ap_adj, roc_feat, ap_feat))

    print("using the latest embeddings to predict")
    for t in range(train_len-1, time_length):
        adj_t_prob = adj_last
        feature_t_prob = features_last
        roc_adj, ap_adj = get_roc_score_adj(test_adjs[t], test_adjs_negative[t], adj_t_prob, t, adjs_ori)
        roc_feat, ap_feat = get_roc_score_feat(test_feas[t], test_feas_false[t], feature_t_prob, t, feats_ori)
        print(" roc_adj:{:.5f} ap_adj:{:.5f}"
              " roc_fea:{:.5f} ap_adj:{:.5f} ".format(roc_adj, ap_adj, roc_feat, ap_feat))
    print("finish testing")
    print("="*30)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model arguments")
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="hidden size of gcn")
    parser.add_argument("--pre-hidden-size", type=int, default=512,
                        help="hidden size of preprocess")
    parser.add_argument("--pre-out-size", type=int, default=512,
                        help="output size of preprocess")
    parser.add_argument("--belief-size", type=int, default=512,
                        help="belief size")
    parser.add_argument("--emb-size", type=int, default=64,
                        help="embedding size")
    parser.add_argument("--decoder-hidden", type=int, default=256,
                        help="hidden size of decoder")
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str, default="DBLP_sub",
                        help='The used dataset name')
    parser.add_argument('--pre_len', type=int, default=1,
                        help="how many time steps to predict")
    parser.add_argument('--co_embedding', type=bool, default=True,
                        help="whether co-embedding")
    args = parser.parse_args()
    print(args)
    train(args)




