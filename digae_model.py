from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from digae_layers import InnerProductDecoder, DirectedInnerProductDecoder, DirectedGCNConvEncoder

import inner_embedding_network


from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import global_sort_pool
from torch_geometric.nn.aggr import GraphMultisetTransformer

EPS = 1e-15
MAX_LOGSTD = 10


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class DirectedGAE(torch.nn.Module):
    def __init__(self, encoder, decoder=None):
        super(DirectedGAE, self).__init__()
        self.encoder = encoder
        self.decoder = DirectedInnerProductDecoder() if decoder is None else decoder

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        s, t = self.encoder(x, x, edge_index)
        adj_pred = self.decoder.forward_all(s, t)
        return adj_pred

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(s, t, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, s, t, pos_edge_index, neg_edge_index):
        # XXX
        pos_y = s.new_ones(pos_edge_index.size(1))
        neg_y = s.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class OneHotDirectedGAE(torch.nn.Module):
    def __init__(self, initial_encoder, embedding_dim, hidden_dim, out_dim, encoder=None, decoder=None):
        super(OneHotDirectedGAE, self).__init__()
        self.encoder = DirectedGCNConvEncoder(embedding_dim, hidden_dim, out_dim, alpha=0.2, beta=0.8,
                                            self_loops=True,
                                            adaptive=False) if encoder is None else encoder
        self.decoder = DirectedInnerProductDecoder() if decoder is None else decoder
        self.initial_encoder = initial_encoder

    def forward(self, data):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x, edge_index = data.x, data.edge_index

        x = self.initial_encoder(x)

        s, t = self.encoder(x, x, edge_index)
        adj_pred = self.decoder.forward_all(s, t)
        return adj_pred

    def encode(self, u,v, edge_index):

        u = self.initial_encoder(u)
        v = self.initial_encoder(v)

        return self.encoder(u,v, edge_index)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode_and_pool(self, u,v, edge_index, batch):
        u = self.initial_encoder(u)
        v = self.initial_encoder(v)

        s,t = self.encoder(u,v,edge_index)

        #s_pool = gmp(s, batch)

        #t_pool = gmp(t, batch)

        s_pool = torch.cat([gmp(s, batch), gap(s, batch)], dim=1)
        t_pool = torch.cat([gmp(t, batch), gap(t, batch)], dim=1)

        # s_pool = global_sort_pool(s, batch, k=3)
        # t_pool = global_sort_pool(t, batch, k=3)



        return torch.cat([s_pool, t_pool], dim = 1)







    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(s, t, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, s, t, pos_edge_index, neg_edge_index):
        # XXX
        pos_y = s.new_ones(pos_edge_index.size(1))
        neg_y = s.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class AttentionDIGAE(torch.nn.Module):
    def __init__(self, initial_encoder, encoder, decoder=None):
        super(AttentionDIGAE, self).__init__()
        self.encoder = encoder
        self.decoder = DirectedInnerProductDecoder() if decoder is None else decoder
        self.initial_encoder = initial_encoder
        self.GMT_pool_layer_s = GraphMultisetTransformer(256, 128, 256, num_nodes = 15)#, pool_sequences = ['SelfAtt'])
        self.GMT_pool_layer_t = GraphMultisetTransformer(256, 128, 256, num_nodes = 15)#, pool_sequences = ['SelfAtt'])

    def forward(self, data):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x, edge_index = data.x, data.edge_index

        x = self.initial_encoder(x)

        s, t = self.encoder(x, x, edge_index)
        adj_pred = self.decoder.forward_all(s, t)
        return adj_pred

    def encode(self, u,v, edge_index):

        u = self.initial_encoder(u)
        v = self.initial_encoder(v)

        return self.encoder(u,v, edge_index)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode_and_pool(self, u,v, edge_index, batch):
        u = self.initial_encoder(u)
        v = self.initial_encoder(v)

        s,t = self.encoder(u,v,edge_index)

        #
        #
        # s_pool = torch.cat([gmp(s, batch), gap(s, batch)], dim=1)
        # t_pool = torch.cat([gmp(t, batch), gap(t, batch)], dim=1)

        s_pool = self.GMT_pool_layer_s(s, index = batch, edge_index=edge_index)
        t_pool = self.GMT_pool_layer_t(t, index = batch, edge_index = edge_index)

        print (s_pool.shape)
        return torch.cat([s_pool,t_pool], dim = 1)




    def recon_loss(self, s, t, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(s, t, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, s.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(s, t, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, s, t, pos_edge_index, neg_edge_index):
        # XXX
        pos_y = s.new_ones(pos_edge_index.size(1))
        neg_y = s.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(s, t, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(s, t, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
