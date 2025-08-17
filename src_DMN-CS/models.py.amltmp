import torch
from torch import nn
from src.embed import DataEmbedding
from src.mha import MultiHeadAttention


class SLP(nn.Module):
    def __init__(self, n_features, n_assets, dim_out, timesteps, **kwargs):
        super().__init__()
        self.layer = nn.Linear(n_features*n_assets*timesteps, dim_out)
        self.timesteps = timesteps
        self.dim_out = dim_out

    def forward(self, x):
        x = x.flatten(start_dim=1) # puts timesteps, n_features and n_assets on the same dim
        out = self.layer(x)
        out = out.view(out.shape[0], self.dim_out)
        out = torch.tanh(out)
        return out


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.layer1     = nn.Linear(d_model, d_model)
        self.layer2     = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=8, n_heads=2, dropout=0.0, multiplier=1):
        super(EncoderLayer, self).__init__()

        assert d_model % multiplier == 0, "d_model must be divisible by multiplier"

        self.ffn            = FFN(d_model=d_model)
        self.dropout        = nn.Dropout(p=dropout)
        self.norm1          = nn.LayerNorm(normalized_shape=d_model)
        self.norm2          = nn.LayerNorm(normalized_shape=d_model)
        self.multihead_attn = MultiHeadAttention(
                                E_q=d_model, E_k=d_model, E_v=d_model,
                                E_total=int(n_heads*d_model/multiplier),
                                nheads=n_heads, dropout=dropout
                                )

    def forward(self, x):

        x_new       = self.multihead_attn(query=x, key=x, value=x)
        x           = x + self.dropout(x_new)
        x           = self.norm1(x)

        x_new       = self.ffn(x)
        x           = x + self.dropout(x_new)
        x           = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, n_enc_layers=1, n_heads=2, d_model=8, dropout=0.0, multiplier=1):
        super().__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = n_heads

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads,
                         dropout=dropout, multiplier=multiplier) 
            for _ in range(n_enc_layers)])

    def forward(self, x):

        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=8, n_heads=2, dropout=0.0, multiplier=1):
        super(DecoderLayer, self).__init__()
        
        assert d_model % multiplier == 0, "d_model must be divisible by multiplier"

        self.ffn            = FFN(d_model=d_model)
        self.dropout        = nn.Dropout(p=dropout)
        self.norm1          = nn.LayerNorm(normalized_shape=d_model)
        self.norm2          = nn.LayerNorm(normalized_shape=d_model)
        self.cross_attn     = MultiHeadAttention(
                                E_q=d_model, E_k=d_model, E_v=d_model, 
                                E_total=int(n_heads*d_model/multiplier),
                                nheads=n_heads, dropout=dropout
                                )

    def forward(self, x_dec, x_enc):

        x_dec_new       = self.cross_attn(query=x_dec, key=x_enc, value=x_enc)
        x_dec           = x_dec + self.dropout(x_dec_new)
        x_dec           = self.norm1(x_dec)

        x_dec_new       = self.ffn(x_dec)
        x_dec           = x_dec + self.dropout(x_dec_new)
        x_dec           = self.norm2(x_dec)

        return x_dec


class Decoder(nn.Module):
    def __init__(self, n_dec_layers=1, n_heads=2, d_model=8, dropout=0.0, multiplier=1):
        super().__init__()

        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = n_heads

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads,
                         dropout=dropout, multiplier=multiplier)
            for _ in range(n_dec_layers)])

    def forward(self, x_dec, x_enc):

        for dec_layer in self.dec_layers:
            x_dec = dec_layer(x_dec, x_enc)

        return x_dec


class Transformer(nn.Module):
    def __init__(self, n_features: int=None, n_assets: int=None, dim_out: int=None, 
                 d_model: int=None, n_enc_layers: int=None, n_dec_layers: int=None, 
                 n_heads: int=None, cat_info: dict=None, dropout: float=0.0, 
                 multiplier: int=1, **kwargs):
        '''
        emb_dim = d_model <-> the dimension that Q and W_q are multiplied across
        '''
        super().__init__()

        self.cat_info = {} if cat_info is None else cat_info
        self.n_features = n_features
        self.n_assets = n_assets
        self.dim_out = dim_out
        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = n_heads

        self.enc_embedding = DataEmbedding(n_features=n_features, n_assets=n_assets, 
                                           d_model=d_model, cat_info=cat_info, 
                                           dropout=dropout)
        self.dec_embedding = DataEmbedding(n_features=n_features, n_assets=n_assets, 
                                           d_model=d_model, cat_info=cat_info, 
                                           dropout=dropout)
        self.encoder = Encoder(n_enc_layers=n_enc_layers, n_heads=n_heads,
                               d_model=d_model, dropout=dropout,
                               multiplier=multiplier)
        self.decoder = Decoder(n_dec_layers=n_dec_layers, n_heads=n_heads,
                               d_model=d_model, dropout=dropout,
                               multiplier=multiplier)
        self.dense = nn.Linear(d_model, dim_out)

    def data_prep(self, x):
        '''Prepares data for the transformer.
        Args:
            x: X_train / X_val / X_test of shape (batchsize, timesteps, n_features, n_assets) (required).
        Values:
            x_enc: continuous features for encoder, of shape (batchsize, timesteps-1, n_assets, n_features)
            x_dec: continuous features for decoder, of shape (batchsize, 1, n_assets, n_features)
            x_cat_enc: categorical features for encoder, of shape (batchsize, timesteps-1, n_cat_features)
            x_cat_dec: categorical features for decoder, of shape (batchsize, 1, n_cat_features)
        '''
        x = x.permute(0, 1, 3, 2) # (batchsize, timesteps, n_assets, n_features)

        x_cont  = x[...,:-len(self.cat_info)] # extract continuous features
        x_cat   = x[...,-len(self.cat_info):] # extract categorical features

        x_enc       = x_cont[:,:-1,:,:] # extract from timestep -1 downwards
        x_dec       = x_cont[:,-1:,:,:] # extract timestep 0

        x_cat_enc   = x_cat[:,:-1,0,:]
        x_cat_dec   = x_cat[:,-1:,0,:] # any asset is good, categorical features are only w.r.t. batch and time

        return x_enc, x_dec, x_cat_enc, x_cat_dec

    def forward(self, x): # (batchsize, timesteps, n_features, n_assets)

        batch_size = x.shape[0]

        assert x.shape[-1] == self.n_assets and x.shape[-2] == self.n_features+len(self.cat_info)
        assert x.dim() == 3 or x.dim() == 4 # ensure either 3D unbatched, or 4D unbatched

        x_enc, x_dec, x_cat_enc, x_cat_dec = self.data_prep(x)

        ## Encoder
        x_enc = self.enc_embedding(x_enc, x_cat_enc) # (batchsize, timesteps, n_assets, d_model)

        x_enc = x_enc.unsqueeze(-2) # (batchsize, timesteps, n_assets, 1, d_model)
        x_enc = x_enc.expand(-1,-1,-1,self.n_assets,-1) # (batchsize, timesteps, n_assets, n_assets, d_model)
        x_enc = x_enc.permute(0,2,1,3,4) # (batchsize, n_assets, timesteps, n_assets, d_model)
        x_enc = x_enc.reshape(
            (
                x_enc.shape[0] if x.dim() == 4 else 1) * self.n_assets,
                x_enc.shape[-3] * self.n_assets,
                self.d_model
            ) # (batchsize x n_assets, timesteps x n_assets, d_model)
        # x_enc = x_enc.permute(0,2,1,3) # (batchsize, n_assets, timesteps, d_model) # TODO: eliminate
        # x_enc = x_enc.flatten(start_dim=0, end_dim=-3) # (batchsize x n_assets, timesteps, d_model) # TODO: eliminate
        x_enc = self.encoder(x_enc) # (batchsize x n_assets, timesteps x n_assets, d_model)

        ## Decoder
        x_dec = self.dec_embedding(x_dec, x_cat_dec) # (batchsize, 1, n_assets, d_model)

        # x_dec = x_dec.unsqueeze(-2) # (batchsize, 1, n_assets, 1, d_model)
        # x_dec = x_dec.expand(-1,-1,-1,self.n_assets,-1) # (batchsize, 1, n_assets, n_assets, d_model)
        # x_dec = x_dec.permute(0,2,1,3,4) # (batchsize, n_assets, 1, n_assets, d_model)
        # x_dec = x_dec.reshape(
                # (x_dec.shape[0] if x.dim() == 4 else 1) * self.n_assets,
                # x_dec.shape[-3] * self.n_assets,
                # self.d_model
            # ) # (batchsize x n_assets, 1 x n_assets, d_model)
        # x = self.decoder(x_dec, x_enc) # (batchsize x n_assets, 1 x n_assets, d_model)
        # idx = torch.tensor([[i*self.n_assets**2, 3+i*self.n_assets**2] for i in range(batch_size)]).flatten()
        # x = x.flatten(end_dim=-2) # (batchsize x n_assets x n_assets, d_model)
        # x = torch.index_select(x, dim=0, index=idx) # (batchsize x n_assets, d_model)
        # x = self.dense(x) # (batchsize x n_assets, 1)
        # x = x.flatten() # (batchsize x n_assets)
        # x = torch.tanh(x) # (batchsize x n_assets)

        x_dec = x_dec.permute(0,2,1,3) # (batchsize, n_assets, 1, d_model) # TODO: eliminate
        x_dec = x_dec.flatten(start_dim=0, end_dim=-3) # (batchsize x n_assets, 1, d_model) # TODO: eliminate
        x = self.decoder(x_dec, x_enc) # (batchsize x n_assets, 1, d_model)

        x = self.dense(x) # (batchsize x n_assets, 1, 1)
        x = x.flatten() # (batchsize x n_assets)
        x = torch.tanh(x) # (batchsize x n_assets)

        return x # (batchsize x n_assets, dim_out)




