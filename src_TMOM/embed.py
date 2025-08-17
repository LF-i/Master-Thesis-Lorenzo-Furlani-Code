import math
import torch
import torch.nn as nn

### From Informer Paper & https://github.com/pytorch/examples/tree/main/word_language_model

class PositionalEmbedding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the token position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        max_len: the max. length of the incoming sequence.
    """
    def __init__(self, d_model: int, max_len: int=1000):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: features tensor, either 3D (batchsize, timesteps, n_assets, n_features)
                or 2D (timesteps, n_assets, n_features).
        Values:
            Positional embedding tensor, 2D as of (timesteps, d_model). 
        '''
        # Adapt length w.r.t. timesteps back of features tensor
        return self.pe[:x.size(-3), :] # (timesteps, d_model)


class EntityEmbedding(nn.Module):
    '''Entity embedding of assets.
    '''
    def __init__(self, n_assets: int, d_model: int):
        super(EntityEmbedding, self).__init__()

        self.embedder = nn.Embedding(n_assets, d_model)
        self.register_buffer('ent_idx', torch.arange(n_assets)) # (n_assets)

    def forward(self): # (n_assets) -> (n_assets, d_model) -> (1, 1, n_assets, d_model)

        ent_tensor = self.embedder(self.ent_idx) # (n_assets, d_model)
        ent_emb = ent_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, n_assets, d_model) <=> 
                                                            # (batchsize, timesteps, n_assets, d_model)
        return ent_emb # (1, 1, n_assets, d_model)


class TokenEmbedding(nn.Module):
    '''Linear projection of scalar context x_i^t into d_model-dim u_i^t.
    '''
    def __init__(self, n_features: int, d_model: int):
        super(TokenEmbedding, self).__init__()

        self.scalar_projection = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: tensor of CONTINUOUS features, either 3D (batchsize, timesteps, n_assets x n_features + n_shared_cols)
                or 2D (timesteps, n_assets, n_features + n_shared_cols).
        Values:
            Tensor with n_assets as penultimate dimension and d_model as last dimension.
        '''
        scaled_x = self.scalar_projection(x) # scalar projection of m features to d_model
        return scaled_x # (batchsize, timesteps, d_model)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super(FixedEmbedding, self).__init__()

        if d_model % 2 != 0:
            raise ValueError('d_model must be even')

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    '''Each global time step is employed by a learnable stamp embeddings.
    '''
    def __init__(self, d_model: int, cat_info: dict, embed_type: str='learnable'):
        super(TemporalEmbedding, self).__init__()

        if embed_type == 'fixed':
            self.embedder = nn.ModuleList([FixedEmbedding(n_unique, d_model) \
                for n_unique in cat_info.values()])
        else:
            self.embedder = nn.ModuleList([nn.Embedding(n_unique, d_model) \
                if key != 'year' else nn.Embedding(n_unique+1, d_model)
                for key, n_unique in cat_info.items()])

        self.max_year = float('inf')
        self.year_idx = None
        i = 0

        for key, n_unique in cat_info.items():
            if key == 'year':
                self.max_train_year = n_unique
                self.year_idx = i
            i += 1

    def forward(self, x_temp: torch.Tensor):
        '''
        Args:
            x: tensor of CATEGORICAL features, either 3D (batch size, timesteps, n_cat_features)
                or 2D (timesteps, n_cat_features).
        '''
        x_temp = x_temp.long()
        embeddings = []

        for i in range(x_temp.shape[-1]): # for every cat feature (batchsize, timesteps, 1)

            if i != self.year_idx:
                embedding = self.embedder[i](x_temp[...,i]) # feed into its embedding layer
                embeddings.append(embedding)                # (batchsize, timesteps, d_model)
                # for each iteration embed the cat feature over all batch elements and timesteps

            else:
                x = torch.clamp(x_temp[...,i], max=self.max_train_year)
                embedding = self.embedder[i](x) # feed into its embedding layer
                embeddings.append(embedding)                # (batchsize, timesteps, d_model)
                # for each iteration embed the cat feature over all batch elements and timesteps

                                                         # stack & sum across cat features
        return torch.stack(embeddings, dim=-1).sum(dim=-1) # (batchsize, timesteps, d_model)


class DataEmbedding(nn.Module):
    """Create embedded features w.r.t. token embedding, position embedding and temporal embedding.
    Args:
        n_features: nr. of continuous features (required).
        n_assets: nr. of assets (required.)
        d_model: the embedding dim (required).
        cat_info: dict mapping categorical vars to their unique values count (required).
        embed_type: 'learnable' / 'fixed', type of temperal embedding.
        dropout: the dropout value.
    """
    def __init__(self, n_features: int, n_assets: int, d_model: int=8, cat_info: dict={}, 
                 embed_type: str='learnable', dropout: float=0.0):
        super(DataEmbedding, self).__init__()

        self.n_assets = n_assets
        self.n_features = n_features
        self.val_embedding = TokenEmbedding(n_features=n_features, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.tem_embedding = TemporalEmbedding(d_model=d_model, cat_info=cat_info, embed_type=embed_type)
        self.ent_embedding = EntityEmbedding(n_assets=n_assets, d_model=d_model) if n_assets > 1 else 0
        self.dropout       = nn.Dropout(p=dropout) # as in Informer

    def forward(self, x: torch.Tensor, x_temp: torch.Tensor):
        '''
        Args:
            x: tensor of CONTINUOUS features, either 3D (batchsize, timesteps, n_assets, n_features)
                or 2D (timesteps, n_assets, n_features).
            x_temp: tensor of CATEGORICAL features, either 3D (batchsize, timesteps, n_cat_features)
                or 2D (timesteps, n_cat_features).
        Values: 
            x: tensor of 3D (batchsize, timesteps, n_assets, d_model)
                or 2D (timesteps, n_assets, d_model)
        '''
        assert self.n_features == x.shape[-1] and self.n_assets == x.shape[-2]

        ## Scalar projection of features to d_model
        val = self.val_embedding(x) # (batchsize, timesteps, n_assets, d_model)

        ## Learnable year and month embeddings 
        tem = self.tem_embedding(x_temp).unsqueeze(-2).expand(-1,-1,self.n_assets,-1) # (batchsize, timesteps, d_model) -> (batchsize, timesteps, n_assets, d_model)

        ## Local context (i.e. the timestamp)
        pos = self.pos_embedding(x).unsqueeze(-2).unsqueeze(0).expand(-1,-1,self.n_assets,-1) # (timesteps, d_model) -> (1, timesteps, n_assets, d_model)

        ## Assets' entity embeddings
        ent = self.ent_embedding() if self.n_assets > 1 else self.ent_embedding # (1, 1, n_assets, d_model)

        x = val + tem + pos + ent
        # (batchsize, timesteps, n_assets, d_model) + (batchsize, timesteps, 1, d_model) + (1, timesteps, 1, d_model) + (1, 1, n_assets, d_model)

        return self.dropout(x) # (batchsize, timesteps, n_assets, d_model)





####################################################################################################
####################################################################################################
### APPENDIX


# class TimeFeatureEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='timeF', freq='h'):
#         super(TimeFeatureEmbedding, self).__init__()

#         freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
#         d_inp = freq_map[freq]
#         self.embed = nn.Linear(d_inp, d_model)
    
#     def forward(self, x):
#         return self.embed(x)


# # "We use the standard positional encoding for day."
# class _PositionalEmbedding(nn.Module):
#     r"""Inject some information about the relative or absolute position of the tokens in the sequence.
#         The positional encodings have the same dimension as the embeddings, so that the two can be summed.
#         Here, we use sine and cosine functions of different frequencies.
#     .. math:
#         \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
#         \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#         \text{where pos is the word position and i is the embed idx)
#     Args:
#         d_model: the embed dim (required).
#         dropout: the dropout value (default=0.1).
#         max_len: the max. length of the incoming sequence (default=5000).
#     Examples:
#         >>> pos_encoder = PositionalEncoding(d_model)
#     """

#     def __init__(self, d_model, dropout=0.0, max_len=1000):
#         super(_PositionalEmbedding, self).__init__()

#         if d_model % 2 != 0:
#             raise ValueError('d_model must be even')

#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term) # even columns
#         pe[:, 1::2] = torch.cos(position * div_term) # odd columns
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [sequence length, batch size, embed dim]
#             output: [sequence length, batch size, embed dim]
#         Examples:
#             >>> output = pos_encoder(x)
#         """
#         # Adds a size-adapted pe to every element of dim-0
#         x = x + self.pe[:x.size(1), :] # adapt pe to the length of x
#         return self.dropout(x)
