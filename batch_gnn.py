import torch
from digae_layers import DirectedGCNConvEncoder, DirectedInnerProductDecoder, SingleLayerDirectedGCNConvEncoder
from digae_model import OneHotDirectedGAE
import json
import inner_embedding_network
from torch_geometric.data import Data
import pickle
from ast_def import *
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader


with open("dep_data.json") as fp:
    dep_db = json.load(fp)
    
with open("new_db.json") as fp:
    new_db = json.load(fp)

#with open("polished_dict.json") as f:
#    p_d = json.load(f)

full_db = {}
count = 0
for key in new_db.keys():
    val = new_db[key]

    if key[0] == " ":
        full_db[key[1:]] = val
    else:
        full_db[key] = val

deps = {}
for key in dep_db.keys():
    val = dep_db[key]

    if key[0] == " ":
        deps[key[1:]] = val
    else:
        deps[key] = val

with open("torch_graph_dict.pk", "rb") as f:
    torch_graph_dict = pickle.load(f)

with open("one_hot_dict.pk", "rb") as f:
    one_hot_dict = pickle.load(f)

with open("train_test_data.pk", "rb") as f:
    train, val, test, enc_nodes = pickle.load(f)

polished_goals = []
for val_ in new_db.values():
    polished_goals.append(val_[2])

tokens = list(set([token.value for polished_goal in polished_goals for token in polished_to_tokens_2(polished_goal)  if token.value[0] != 'V']))

tokens.append("VAR")
tokens.append("VARFUNC")
tokens.append("UNKNOWN")


class LinkData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, edge_attr_s=None, edge_attr_t = None,
                 y=None, x_s_one_hot=None, x_t_one_hot=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t
        self.x_s_one_hot=x_s_one_hot
        self.x_t_one_hot=x_t_one_hot
        self.y = y
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# new_train = []
#
# for (x1, x2, y) in train:
#     x1_graph = torch_graph_dict[x1]
#     x2_graph = torch_graph_dict[x2]
#
#     new_train.append(LinkData(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1]))
#
# new_val = []
#
# for (x1, x2, y) in val:
#     x1_graph = torch_graph_dict[x1]
#     x2_graph = torch_graph_dict[x2]
#
#     new_val.append(LinkData(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1]))


#edge labelled data
new_train = []

for (x1, x2, y) in train:
    x1_graph = torch_graph_dict[x1]
    x2_graph = torch_graph_dict[x2]

    new_train.append(LinkData(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1]))

new_val = []

for (x1, x2, y) in val:
    x1_graph = torch_graph_dict[x1]
    x2_graph = torch_graph_dict[x2]

    new_val.append(LinkData(edge_index_s=x2_graph.edge_index, x_s=x2_graph.x, edge_index_t=x1_graph.edge_index, x_t=x1_graph.x, edge_attr_t=x1_graph.edge_attr, edge_attr_s=x2_graph.edge_attr, y=y, x_s_one_hot=one_hot_dict[x2],  x_t_one_hot=one_hot_dict[x1]))


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))

def loss(graph_net, batch, fc):#, F_p, F_i, F_o, F_x, F_c, conv1, conv2, num_iterations):

    g0_embedding = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))

    g1_embedding = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))

    preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))

    eps = 1e-6

    preds = torch.clip(preds, eps, 1-eps)

    return binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

def accuracy(graph_net, batch, fc):

    g0_embedding = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device),batch.x_t_batch.to(device))

    g1_embedding = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device),batch.x_s_batch.to(device))

    preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)


def run(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):

    loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_net = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)

    fc = inner_embedding_network.F_c_module_(embedding_dim * 8).to(device)

    optimiser_gnn = torch.optim.Adam(list(graph_net.parameters()), lr=step_size, weight_decay=decay_rate)

    optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)

    training_losses = []

    val_losses = []

    for j in range(num_epochs):
        for i, batch in tqdm(enumerate(loader)):

            optimiser_fc.zero_grad()

            optimiser_gnn.zero_grad()

            loss_val = loss(graph_net, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

            loss_val.backward()

            optimiser_fc.step()

            optimiser_gnn.step()

            training_losses.append(loss_val.detach() / batch_size)

            if i % 100 == 0:

                validation_loss = accuracy(graph_net, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                val_losses.append((validation_loss.detach(), j, i))

                val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print ("Val acc: {}".format(validation_loss.detach()))

    #only save encoder for now
    if save == True:
        torch.save(graph_net, "model_checkpoints/gnn_encoder_latest")


    return training_losses, val_losses

def plot_losses(train_loss, val_loss):

    #plt.plot(np.convolve([t[0].cpu().numpy() for t in val_loss], np.ones(40)/40, mode='valid'))
    plt.plot([t[0].cpu().numpy() for t in val_loss])
    plt.show()
    plt.plot(np.convolve([t.cpu().numpy() for t in train_loss], np.ones(1000)/1000, mode='valid'))
    plt.show()












#define setup for separate premise and goal GNNs

def loss_2(graph_net_1, graph_net_2, batch, fc):#, F_p, F_i, F_o, F_x, F_c, conv1, conv2, num_iterations):

    g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))

    g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))

    preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))

    eps = 1e-6

    preds = torch.clip(preds, eps, 1-eps)

    return binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

def accuracy_2(graph_net_1, graph_net_2, batch, fc):

    g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device),batch.x_t_batch.to(device))

    g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device),batch.x_s_batch.to(device))

    preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)


def run_2(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):

    loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_net_1 = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)

    graph_net_2 = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)

    fc = inner_embedding_network.F_c_module_(embedding_dim * 8).to(device)

    optimiser_gnn_1 = torch.optim.Adam(list(graph_net_1.parameters()), lr=step_size, weight_decay=decay_rate)
    optimiser_gnn_2 = torch.optim.Adam(list(graph_net_2.parameters()), lr=step_size, weight_decay=decay_rate)

    optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)

    training_losses = []

    val_losses = []

    for j in range(num_epochs):
        for i, batch in tqdm(enumerate(loader)):

            optimiser_fc.zero_grad()

            optimiser_gnn_1.zero_grad()
            optimiser_gnn_2.zero_grad()

            loss_val = loss_2(graph_net_1,graph_net_2, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

            loss_val.backward()

            optimiser_gnn_1.step()
            optimiser_gnn_2.step()

            optimiser_fc.step()


            training_losses.append(loss_val.detach() / batch_size)

            if i % 100 == 0:

                validation_loss = accuracy_2(graph_net_1, graph_net_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                val_losses.append((validation_loss.detach(), j, i))

                val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print ("Val acc: {}".format(validation_loss.detach()))

    #only save encoder for now
    if save == True:
        torch.save(graph_net_1, "model_checkpoints/gnn_encoder_latest_1")
        torch.save(graph_net_2, "model_checkpoints/gnn_encoder_latest_2")


    return training_losses, val_losses





#run_2(1e-3, 0, 20, 1024, 64, 2, False)

















import gnn_edge_labels


#define setup for separate premise and goal GNNs

def loss_edges(graph_net_1, graph_net_2, batch, fc):#, F_p, F_i, F_o, F_x, F_c, conv1, conv2, num_iterations):

    g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device), batch.edge_attr_t.to(device), batch.x_t_batch.to(device))

    g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device), batch.edge_attr_s.to(device), batch.x_s_batch.to(device))

    preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))

    eps = 1e-6

    preds = torch.clip(preds, eps, 1-eps)

    return binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm

def accuracy_edges(graph_net_1, graph_net_2, batch, fc):

    g0_embedding = graph_net_1(batch.x_t.to(device), batch.edge_index_t.to(device),batch.edge_attr_t.to(device), batch.x_t_batch.to(device))

    g1_embedding = graph_net_2(batch.x_s.to(device), batch.edge_index_s.to(device), batch.edge_attr_s.to(device), batch.x_s_batch.to(device))

    preds = fc(torch.cat([g0_embedding, g1_embedding], axis=1))

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)


def run_edges(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):

    loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # graph_net_1 = gnn_edge_labels.message_passing_gnn_edges(len(tokens), embedding_dim, graph_iterations, device)
    #
    # graph_net_2 = gnn_edge_labels.message_passing_gnn_edges(len(tokens), embedding_dim, graph_iterations, device)

    # graph_net_1 = gnn_edge_labels.message_passing_gnn_edges_gine(len(tokens), embedding_dim, graph_iterations, device)
    #
    # graph_net_2 = gnn_edge_labels.message_passing_gnn_edges_gine(len(tokens), embedding_dim, graph_iterations, device)

    fc = gnn_edge_labels.F_c_module_(embedding_dim * 8).to(device)

    optimiser_gnn_1 = torch.optim.Adam(list(graph_net_1.parameters()), lr=step_size, weight_decay=decay_rate)
    optimiser_gnn_2 = torch.optim.Adam(list(graph_net_2.parameters()), lr=step_size, weight_decay=decay_rate)

    optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)

    training_losses = []

    val_losses = []

    for j in range(num_epochs):
        for i, batch in tqdm(enumerate(loader)):

            optimiser_fc.zero_grad()

            optimiser_gnn_1.zero_grad()
            optimiser_gnn_2.zero_grad()

            loss_val = loss_edges(graph_net_1,graph_net_2, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

            loss_val.backward()

            optimiser_gnn_1.step()
            optimiser_gnn_2.step()

            optimiser_fc.step()


            training_losses.append(loss_val.detach() / batch_size)

            if i % 100 == 0:

                validation_loss = accuracy_edges(graph_net_1, graph_net_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                val_losses.append((validation_loss.detach(), j, i))

                val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print ("Val acc: {}".format(validation_loss.detach()))

    #only save encoder for now
    if save == True:
        torch.save(graph_net_1, "model_checkpoints/gnn_encoder_latest_1")
        torch.save(graph_net_2, "model_checkpoints/gnn_encoder_latest_2")


    return training_losses, val_losses


#run_edges(1e-3, 0, 20, 1024, 64, 0, False)
#run_2(1e-3, 0, 20, 1024, 64, 4, False)

def accuracy_digae(model_1, model_2, batch, fc):

    data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device))
    data_2 = Data(x=batch.x_s.to(device), edge_index=batch.edge_index_s.to(device))

    # g0_embedding = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))
    #
    # g1_embedding = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))

    u1 = data_1.x.clone().to(device)
    v1 = data_1.x.clone().to(device)

    train_pos_edge_index_1 = data_1.edge_index.clone().to(device)

    u2 = data_2.x.clone().to(device)
    v2 = data_2.x.clone().to(device)

    train_pos_edge_index_2 = data_2.edge_index.clone().to(device)

    graph_enc_1 = model_1.encode_and_pool(u1, v1, train_pos_edge_index_1, batch.x_t_batch.to(device))

    graph_enc_2 = model_2.encode_and_pool(u2, v2, train_pos_edge_index_2, batch.x_s_batch.to(device))

    preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))

    preds = torch.flatten(preds)

    preds = (preds>0.5).long()

    return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(batch.y)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_digae(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations, save=False):

    loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])

    val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_dim = 64
    out_dim = 64

    initial_encoder = inner_embedding_network.F_x_module_(len(tokens), embedding_dim).to(device)
    decoder = DirectedInnerProductDecoder()

    # encoder = DirectedGCNConvEncoder(embedding_dim, hidden_dim, out_dim, alpha=0.2, beta=0.8,
    #                                         self_loops=True,
    #                                         adaptive=False)

    graph_net_1 = OneHotDirectedGAE(initial_encoder, embedding_dim, hidden_dim, out_dim).to(device)
    graph_net_2 = OneHotDirectedGAE(initial_encoder, embedding_dim, hidden_dim, out_dim).to(device)

    fc = gnn_edge_labels.F_c_module_(embedding_dim * 8).to(device)

    # op_enc =torch.optim.Adam(encoder.parameters(), lr=step_size)
    op_g1 =torch.optim.Adam(graph_net_1.parameters(), lr=step_size)
    op_g2 =torch.optim.Adam(graph_net_2.parameters(), lr=step_size)
    op_fc =torch.optim.Adam(fc.parameters(), lr=step_size)

    training_losses = []

    val_losses = []
    best_acc = 0.

    for j in range(num_epochs):
        print (f"Epoch: {j}")
        for i, batch in tqdm(enumerate(loader)):

            # op_enc.zero_grad()
            op_g1.zero_grad()
            op_g2.zero_grad()
            op_fc.zero_grad()

            data_1 = Data(x=batch.x_t.to(device), edge_index=batch.edge_index_t.to(device))
            data_2 = Data(x = batch.x_s.to(device), edge_index = batch.edge_index_s.to(device))


            u1 = data_1.x.clone().to(device)
            v1 = data_1.x.clone().to(device)


            train_pos_edge_index_1 = data_1.edge_index.clone().to(device)

            u2 = data_2.x.clone().to(device)
            v2 = data_2.x.clone().to(device)

            train_pos_edge_index_2 = data_2.edge_index.clone().to(device)

            graph_enc_1 = graph_net_1.encode_and_pool(u1, v1, train_pos_edge_index_1, batch.x_t_batch.to(device))

            graph_enc_2 = graph_net_2.encode_and_pool(u2, v2, train_pos_edge_index_2, batch.x_s_batch.to(device))


            preds = fc(torch.cat([graph_enc_1, graph_enc_2], axis=1))

            eps = 1e-6

            preds = torch.clip(preds, eps, 1 - eps)

            loss = binary_loss(torch.flatten(preds), torch.LongTensor(batch.y).to(device))

            loss.backward()

            # op_enc.step()
            op_g1.step()
            op_g2.step()
            op_fc.step()

            training_losses.append(loss.detach() / batch_size)

            if i % 100 == 0:

                validation_loss = accuracy_digae(graph_net_1, graph_net_2, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)

                val_losses.append((validation_loss.detach(), j, i))

                val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))

                print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))

                print ("Val acc: {}".format(validation_loss.detach()))

                if validation_loss > best_acc:
                    best_acc = validation_loss
                    print (f"New best validation accuracy: {best_acc}")
                    #only save encoder if best accuracy so far
                    if save == True:
                        torch.save(graph_net_1, "model_checkpoints/gnn_encoder_latest_goal")
                        torch.save(graph_net_2, "model_checkpoints/gnn_encoder_latest_premise")

    print (f"Best validation accuracy: {best_acc}")

    return training_losses, val_losses

run_digae(1e-3, 0, 40, 1024, 64, 2, save=True)


#todo add test set evaluation

##########################################
##########################################

#meta graph

##########################################
##########################################

# num_nodes = len(enc_nodes.get_feature_names_out())
#
# def new_batch_loss(batch, struct_net_target,
#                    struct_net_source,
#                    graph_net, theta1, theta2, theta3,
#                    gamma1, gamma2, b1, b2):
#     B = len(batch)
#
#     def phi_1(x):
#         return inner_embedding_network.phi(x, gamma1, b1)
#
#     def phi_2(x):
#         return inner_embedding_network.phi(x, gamma2, b2)
#
#     x_t_struct = struct_net_target(inner_embedding_network.sp_to_torch(sp.sparse.vstack(batch.x_t_one_hot)).to(device))
#     x_s_struct = struct_net_source(inner_embedding_network.sp_to_torch(sp.sparse.vstack(batch.x_s_one_hot)).to(device))
#
#     x_t_attr = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device))
#
#     x_s_attr = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), batch.x_s_batch.to(device))
#
#
#     # x_t_attr = x_t_attr.reshape(x_t_attr.shape[1],x_t_attr.shape[0])
#     # x_s_attr = x_s_attr.reshape(x_s_attr.shape[1],x_s_attr.shape[0])
#
#     sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#     attr_sim = sim_func(x_t_attr, x_s_attr)
#     struct_sim = sim_func(x_t_struct, x_s_struct)
#     align_sim = sim_func(x_t_attr, x_s_struct)
#
#     # print ((1/32) * sum(batch.y * struct_sim + ((1. - batch.y) * -1. * attr_sim)))
#
#     attr_loss = inner_embedding_network.loss_similarity(attr_sim, batch.y.to(device), phi_1, phi_2)
#     struct_loss = inner_embedding_network.loss_similarity(struct_sim, batch.y.to(device), phi_1, phi_2)
#     align_loss = inner_embedding_network.loss_similarity(align_sim, batch.y.to(device), phi_1, phi_2)
#
#     tot_loss = theta1 * attr_loss + theta2 * struct_loss + theta3 * align_loss
#     return (1. / B) * torch.sum(tot_loss)
#
#
#
#
#
# def run(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations):
#
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     graph_net = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)
#
#     fc = inner_embedding_network.F_c_module_(embedding_dim * 8).to(device)
#
#     optimiser_gnn = torch.optim.Adam(list(graph_net.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     optimiser_fc = torch.optim.Adam(list(fc.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     training_losses = []
#
#     val_losses = []
#
#     for j in range(num_epochs):
#         for i, batch in tqdm(enumerate(loader)):
#
#             optimiser_fc.zero_grad()
#
#             optimiser_gnn.zero_grad()
#
#             loss_val = loss(graph_net, batch, fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#             loss_val.backward()
#
#             optimiser_fc.step()
#
#             optimiser_gnn.step()
#
#             training_losses.append(loss_val.detach() / batch_size)
#
#             if i % 100 == 0:
#
#                 #todo: val moving average, every e.g. 25 record val, then take avg at 1000
#
#                 validation_loss = accuracy(graph_net, next(val_loader), fc)#, fp, fi, fo, fx, fc,conv1,conv2, graph_iterations)
#
#                 val_losses.append((validation_loss.detach(), j, i))
#
#
#                 val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#                 print ("Curr training loss avg: {}".format(sum(training_losses[-100:]) / len(training_losses[-100:])))
#
#                 print ("Val acc: {}".format(validation_loss.detach()))
#
#         #print ("Epoch {} done".format(j))
#
#     return training_losses, val_losses
#
#
#
# def run_meta(step_size, decay_rate, num_epochs, batch_size, embedding_dim, graph_iterations):
#     loader = DataLoader(new_train, batch_size=batch_size, follow_batch=['x_s', 'x_t'])
#
#     val_loader = iter(DataLoader(new_val, batch_size=2048, follow_batch=['x_s', 'x_t']))
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     graph_net = inner_embedding_network.message_passing_gnn_(len(tokens), embedding_dim, graph_iterations, device)
#
#     optimiser_gnn = torch.optim.Adam(list(graph_net.parameters()), lr=step_size, weight_decay=decay_rate)
#
#     struct_net_source = inner_embedding_network.StructureEmbeddingSource(num_nodes, embedding_dim * 2).to(device)
#
#     struct_net_target = inner_embedding_network.StructureEmbeddingTarget(num_nodes, embedding_dim * 2).to(device)
#
#     optimiser_struct_source = torch.optim.Adam(list(struct_net_source.parameters()), lr=step_size,
#                                                weight_decay=decay_rate)
#
#     optimiser_struct_target = torch.optim.Adam(list(struct_net_target.parameters()), lr=step_size,
#                                                weight_decay=decay_rate)
#
#     training_losses = []
#     val_losses = []
#
#     for j in range(num_epochs):
#         for i, batch in tqdm(enumerate(loader)):
#             optimiser_gnn.zero_grad()
#             optimiser_struct_target.zero_grad()
#             optimiser_struct_source.zero_grad()
#
#             theta1 = 0.7
#             theta2 = 0.2
#             theta3 = 0.1
#
#
#             loss_val = new_batch_loss(batch, struct_net_target,
#                                       struct_net_source,
#                                       graph_net, theta1, theta2, theta3,
#                                       gamma1=2, gamma2=2, b1=0.1, b2=0.1)
#
#             loss_val.backward()
#
#             optimiser_gnn.step()
#             optimiser_struct_target.step()
#             optimiser_struct_source.step()
#
#             training_losses.append(loss_val.detach() / batch_size)
#             # print (loss_val.detach())
#
#     return training_losses#, val_losses


# def val_batch(batch, struct_net_target,
#               struct_net_source,
#               graph_net, theta1, theta2, theta3,
#               gamma1, gamma2, b1, b2):
#
#     x_t_struct = struct_net_target(sp_to_torch(sp.sparse.vstack(batch.x_t_one_hot)).to(device))
#     x_s_struct = struct_net_source(sp_to_torch(sp.sparse.vstack(batch.x_s_one_hot)).to(device))
#
#     x_t_attr = graph_net(batch.x_t.to(device), batch.edge_index_t.to(device), batch.x_t_batch.to(device)).to(device)
#
#     x_s_attr = graph_net(batch.x_s.to(device), batch.edge_index_s.to(device), F_p, F_i, F_o, F_x, conv1, conv2,
#                          num_iterations, batch.x_s_batch.to(device)).to(device)
#
#     sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#     attr_sim = sim_func(x_t_attr, x_s_attr)
#     struct_sim = sim_func(x_t_struct, x_s_struct)
#     align_sim = sim_func(x_t_attr, x_s_struct)
#
#     scores = 0.5 * attr_sim + 0.5 * align_sim
#
#     preds = (scores > 0).long()
#
#     return torch.sum(preds == torch.LongTensor(batch.y).to(device)) / len(
#         batch.y)  # best_score(list(zip(scores, batch.y.to(device))))#curr_best_acc, curr_best_lam
#
#
#




    # find scoring threshold which gives the best prediction metric (accuracy for now)
    #     def best_score(scores):
    #         #sort scores
    #         sorted_scores = sorted(scores, key=lambda tup: tup[0])

    #         #print (sorted_scores[:10])

    #         #evaluate metric (accuracy here) using sorted scores and index
    #         def metric_given_threshold(index):
    #             pos = scores[index:]
    #             neg = scores[:index]

    #             correct = len([x for x in pos if x[1] == 1]) + len([x for x in neg if x[1] == 0])

    #             return correct / len(sorted_scores)

    #         #loop through indices testing best threshold
    #         curr_best_metric = 0.
    #         curr_best_idx = 0

    #         for i in range(len(sorted_scores)):
    #             new = metric_given_threshold(i)
    #             if new > curr_best_metric:
    #                 curr_best_metric = new
    #                 curr_best_idx = i

    #         return curr_best_metric, curr_best_idx

    #     #only need one lambda when doing inductive since there's only 2 values to weigh
    #     lam_grid = np.logspace(-1,1,10)

    #     #grid search over lambda for best score

    #     curr_best_lam = 0
    #     curr_best_acc = 0

    #     for lam in lam_grid:
    #         scores = []
    #         for (x1,x2,y) in sims:
    #             scores.append((x1 + lam * x2, y))

    #         acc, idx = best_score(scores)
    #         if acc > curr_best_acc:
    #             curr_best_acc = acc
    #             curr_best_lam = lam

    # keep lambda as thetas for now











