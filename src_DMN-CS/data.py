from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import abc
from copy import deepcopy

num_workers = 0

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainValTestSplitter(abc.ABC):
    @abc.abstractmethod
    def split(self, start, test_delta, seed):
        pass


# data format used in https://arxiv.org/pdf/2302.10175.pdf
class MultivariateTrainValTestSplitter(TrainValTestSplitter):
    def __init__(self, 
                 data: dict[pd.DataFrame],  # features, whose keys are the corresponding assets, 
                 cols: list[str],           # label-speficic features to use
                 cat_cols: list[str],       # datetime features to use
                 target_col: str,           # specify target (volatility scaled daily returns)
                 orig_returns_col: str,     # specify daily returns
                 vol_col: str,              # specify daily ex-ante volatility estimate
                 shared_cols: list[str]=None,
                 timesteps: int=5,         # history size 
                 scaling: str=None,         # None, 'minmax' or 'standard'
                 batch_size: int=100):
        self._data = deepcopy(data)
        self._cols = cols
        self._shared_cols = shared_cols
        self._cat_cols = cat_cols
        self._target_col = target_col
        self._orig_returns_col = orig_returns_col
        self._vol_col = vol_col
        self._scaling = scaling
        self._timesteps = timesteps
        self._batch_size = batch_size

        # To avoid data leak
        assert self._target_col not in self._cols       # ensure y is not in X
        assert self._orig_returns_col not in self._cols # ensure function of y is not in X
        assert self._timesteps > 0
        assert self._batch_size > 0 if self._batch_size != None else False

    def _to_tensorX(self, x: list) -> np.ndarray:
        x = np.stack(x, axis=-1) # concatenate list across assets w.r.t. cols
        x = torch.from_numpy(x).float()
        return x
    def _to_tensor(self, x: list) -> np.ndarray:
        x = np.concatenate(x, axis=-1) # concatenate list across assets w.r.t. cols
        x = torch.from_numpy(x).float()
        return x


    def split_SharedFeats(
            self,
            start: pd.Timestamp,
            test_delta: pd.Timedelta,
            seed: int
        ) -> tuple[DataLoader, DataLoader, DataLoader, pd.Index, dict]:

        offset_delta = pd.Timedelta('1day')

        # Containers for every asset ------------------------------------------------
        X_train, X_val, X_test                  = [], [], []
        y_train, y_val, y_test                  = [], [], []
        y_train_orig, y_val_orig, y_test_orig   = [], [], []
        vol_train, vol_val, vol_test            = [], [], []
        test_datetimes                          = []

        # These will be filled only during the first iteration ----------------------
        first_asset_seen     = False            # flag
        shared_base_df       = None             # df used to build shared / cat features
        train_idx_all = val_idx_all = test_idx_all = None

        # ---------------------------------------------------------------------------
        # (1) Loop over assets – identical to the old routine up to the append phase
        # ---------------------------------------------------------------------------
        for key in self._data.keys():

            features_of_asset = self._data[key].copy()
            features_of_asset['idx'] = np.arange(len(features_of_asset))

            train_val_test = features_of_asset.loc[:start + test_delta]

            # scaling ----------------------------------------------------------------
            if self._scaling is not None:
                if self._scaling == 'minmax':
                    scaler = MinMaxScaler().fit(train_val_test.loc[:start, self._cols])
                elif self._scaling == 'standard':
                    scaler = StandardScaler().fit(train_val_test.loc[:start, self._cols])
                else:
                    raise NotImplementedError
                train_val_test.loc[:, self._cols] = scaler.transform(
                    train_val_test.loc[:, self._cols])

            # tensors ----------------------------------------------------------------
            X   = np.zeros((len(train_val_test), self._timesteps, len(self._cols)))
            y   = np.zeros((len(train_val_test), 1))
            y_o = np.zeros((len(train_val_test), 1))
            vol = np.zeros((len(train_val_test), 1))

            for i, col in enumerate(self._cols):
                for j in range(self._timesteps):
                    X[:, j, i] = train_val_test[col].shift(self._timesteps - j - 1)

            y[:, 0]   = train_val_test[self._target_col]
            y_o[:, 0] = train_val_test[self._orig_returns_col]
            vol[:, 0] = train_val_test[self._vol_col]

            # split indices -----------------------------------------------------------
            train_val_last_idx = train_val_test.loc[:start, 'idx'].iloc[-1]
            start_val          = train_val_test.index[
                train_val_test['idx'] == round(train_val_last_idx * 0.9)][0]

            train_idx = train_val_test.loc[:start_val - offset_delta, 'idx']
            val_idx   = train_val_test.loc[start_val:start - offset_delta, 'idx']
            test_idx  = train_val_test.loc[start:start + test_delta, 'idx']

            # store indices / df only once -------------------------------------------
            if not first_asset_seen:
                train_idx_all = train_idx.to_numpy()
                val_idx_all   = val_idx.to_numpy()
                test_idx_all  = test_idx.to_numpy()
                shared_base_df = train_val_test        # keep for shared / cat feats
                first_asset_seen = True

            test_datetimes.append(
                train_val_test.loc[train_val_test['idx'].isin(test_idx)].index)

            # split sets --------------------------------------------------------------
            X_tr, X_vl, X_te = X[train_idx], X[val_idx], X[test_idx]
            y_tr, y_vl, y_te = y[train_idx], y[val_idx], y[test_idx]
            yo_tr,yo_vl,yo_te= y_o[train_idx],y_o[val_idx],y_o[test_idx]
            v_tr, v_vl, v_te = vol[train_idx],vol[val_idx],vol[test_idx]

            # eliminate leading NaNs --------------------------------------------------
            X_tr       = X_tr[self._timesteps - 1:]
            y_tr       = y_tr[self._timesteps - 1:]
            yo_tr      = yo_tr[self._timesteps - 1:]
            v_tr       = v_tr[self._timesteps - 1:]

            # append to containers ----------------------------------------------------
            X_train.append(X_tr);  X_val.append(X_vl);  X_test.append(X_te)
            y_train.append(y_tr);  y_val.append(y_vl);  y_test.append(y_te)
            y_train_orig.append(yo_tr); y_val_orig.append(yo_vl); y_test_orig.append(yo_te)
            vol_train.append(v_tr);   vol_val.append(v_vl);   vol_test.append(v_te)

        # ---------------------------------------------------------------------------
        # (2) Stack across assets and *flatten* the asset dimension
        # ---------------------------------------------------------------------------
        arrays_X        = [X_train, X_val, X_test]
        arrays_not_X    = [y_train, y_val, y_test,
                           y_train_orig, y_val_orig, y_test_orig,
                           vol_train, vol_val, vol_test]

        X_train, X_val, X_test = list(map(self._to_tensorX, arrays_X))
        y_train, y_val, y_test, y_train_orig, y_val_orig,\
            y_test_orig, vol_train, vol_val, vol_test = list(map(self._to_tensor,
                                                                 arrays_not_X))

        # X_* have shape (N, T, n_features, n_assets) –> flatten last two dims
        def _flatten(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(t.shape[0], t.shape[1], -1)

        X_train, X_val, X_test = map(_flatten, (X_train, X_val, X_test))

        # ---------------------------------------------------------------------------
        # (3) Append *shared* features (no repetition per asset)
        # ---------------------------------------------------------------------------
        if self._shared_cols:
            X_shared = np.zeros((len(shared_base_df), self._timesteps,
                                 len(self._shared_cols)))
            for i, col in enumerate(self._shared_cols):
                for j in range(self._timesteps):
                    X_shared[:, j, i] = shared_base_df[col].shift(
                        self._timesteps - j - 1)

            X_shared = torch.from_numpy(X_shared).float()

            X_train_sh = X_shared[train_idx_all, ...][self._timesteps - 1:]
            X_val_sh   = X_shared[val_idx_all, ...]
            X_test_sh  = X_shared[test_idx_all, ...]

            X_train = torch.cat([X_train, X_train_sh], dim=-1)
            X_val   = torch.cat([X_val,   X_val_sh],   dim=-1)
            X_test  = torch.cat([X_test,  X_test_sh],  dim=-1)

            n_shared = len(self._shared_cols)
        else:
            n_shared = 0

        # ---------------------------------------------------------------------------
        # (4) Append *categorical* / date‑time features (once, no per‑asset copy)
        # ---------------------------------------------------------------------------
        if self._cat_cols:
            X_cat = np.zeros((len(shared_base_df), self._timesteps,
                              len(self._cat_cols)))
            for i, col in enumerate(self._cat_cols):
                for j in range(self._timesteps):
                    X_cat[:, j, i] = shared_base_df[col].shift(
                        self._timesteps - j - 1)

            X_cat = torch.from_numpy(X_cat).float()

            X_train_cat = X_cat[train_idx_all, ...][self._timesteps - 1:]
            X_val_cat   = X_cat[val_idx_all, ...]
            X_test_cat  = X_cat[test_idx_all, ...]

            X_train = torch.cat([X_train, X_train_cat], dim=-1)
            X_val   = torch.cat([X_val,   X_val_cat],   dim=-1)
            X_test  = torch.cat([X_test,  X_test_cat],  dim=-1)

            cat_info = {col_name: len(torch.unique(X_train_cat[..., i])) # TODO: X_train_cat
                        for i, col_name in enumerate(self._cat_cols)}
        else:
            cat_info = {}

        # ---------------------------------------------------------------------------
        # (5) Build DataLoaders
        # ---------------------------------------------------------------------------
        for i in range(1, len(test_datetimes)):
            assert np.all(
                (test_datetimes[i] - test_datetimes[0]) == pd.Timedelta(0))
        assert len(X_test) == len(test_datetimes[0])

        g = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(TensorDataset(X_train, y_train,
                                                y_train_orig, vol_train),
                                  shuffle=True, batch_size=self._batch_size,
                                  worker_init_fn=seed_worker,
                                  generator=g, num_workers=num_workers)

        val_loader   = DataLoader(TensorDataset(X_val,   y_val,
                                                y_val_orig,   vol_val),
                                  shuffle=False, batch_size=self._batch_size,
                                  num_workers=num_workers)

        test_loader  = DataLoader(TensorDataset(X_test,  y_test,
                                                y_test_orig,  vol_test),
                                  shuffle=False, batch_size=self._batch_size,
                                  num_workers=num_workers)

        params = (len(self._cols), len(self._data.keys()), n_shared) # n_features, n_assets, n_shared

        return train_loader, val_loader, test_loader, test_datetimes[0], cat_info, params


    def split(self,
              start: pd.Timestamp,      # end of training / start of valuation 
              test_delta: pd.Timedelta, # testing length of span
              seed: int
              ) -> tuple[DataLoader, DataLoader, DataLoader, pd.Index, dict]:     
        '''
        Data is splitted into training, validation and testing datasets.
        Training ends at start-0.1*start, validation ends at start, testing ends at start+test_delta.
        '''
        offset_delta = pd.Timedelta('1day') # used to avoid overlap of train/val/test in pandas .loc

        X_train, X_val, X_test                  = [], [], []
        y_train, y_val, y_test                  = [], [], []
        y_train_orig, y_val_orig, y_test_orig   = [], [], []
        vol_train, vol_val, vol_test            = [], [], []
        test_datetimes                          = []

        ####################################################################################################
        ### Start: Loop for Every Asset ####################################################################

        for key in self._data.keys(): # for every asset

            features_of_asset = self._data[key].copy()
            features_of_asset['idx'] = np.arange(len(features_of_asset)) # add index

            train_val_test = features_of_asset.loc[:start+test_delta]  # subset in time

            # Scale features
            if self._scaling is not None:
                if self._scaling == 'minmax':
                    scaler = MinMaxScaler().fit(train_val_test.loc[:start, self._cols]) # fit scaling model on training data
                elif self._scaling == 'standard':
                    scaler = StandardScaler().fit(train_val_test.loc[:start, self._cols]) # fit scaling model on training data
                else:
                    raise NotImplementedError

                train_val_test.loc[:, self._cols] = scaler.transform(train_val_test.loc[:, self._cols]) # scale all data

            # Create storage
            X       = np.zeros((len(train_val_test), self._timesteps, len(self._cols)))
            y       = np.zeros((len(train_val_test), 1))
            y_orig  = np.zeros((len(train_val_test), 1)) # collect non scaled target returns data for further model evaluation
            vol     = np.zeros((len(train_val_test), 1)) # collect volatility data for turnover regularization and/or evaluation

            # Fill X
            for i, col in enumerate(self._cols): # for every column
                for j in range(self._timesteps): # for every timestep
                    X[:, j, i] = train_val_test[col].shift(self._timesteps - j - 1) # from today to (timesteps-1) days back

            y[:,0]      = train_val_test[self._target_col]
            y_orig[:,0] = train_val_test[self._orig_returns_col]
            vol[:,0]    = train_val_test[self._vol_col]

            ## Train/val/test split
            train_validation_idx = train_val_test.loc[:start, 'idx'].iloc[-1]
            start_val = train_val_test.index[train_val_test['idx'] == round(train_validation_idx * 0.9)][0]

            # Get index
            train_idx   = train_val_test.loc[:start_val-offset_delta, 'idx']
            val_idx     = train_val_test.loc[start_val:start-offset_delta, 'idx']
            test_idx    = train_val_test.loc[start:start+test_delta, 'idx']
            # NB: there is overlap equal to self._timesteps between folds

            # Get testing dates
            test_dt     = train_val_test.loc[train_val_test['idx'].isin(test_idx)].index
            test_datetimes.append(test_dt)
            
            # Split
            X_train_, y_train_, y_train_orig_, vol_train_   = X[train_idx], y[train_idx], y_orig[train_idx], vol[train_idx]
            X_val_, y_val_, y_val_orig_, vol_val_           = X[val_idx], y[val_idx], y_orig[val_idx], vol[val_idx]
            X_test_, y_test_, y_test_orig_, vol_test_       = X[test_idx], y[test_idx], y_orig[test_idx], vol[test_idx]

            # Remove NaN samples after applying pd.Series.shift
            X_train_        = X_train_[self._timesteps-1:]
            y_train_        = y_train_[self._timesteps-1:]
            y_train_orig_   = y_train_orig_[self._timesteps-1:]
            vol_train_      = vol_train_[self._timesteps-1:]

            X_train.append(X_train_)
            X_val.append(X_val_)
            X_test.append(X_test_)

            y_train.append(y_train_)
            y_val.append(y_val_)
            y_test.append(y_test_)

            y_train_orig.append(y_train_orig_)
            y_val_orig.append(y_val_orig_)
            y_test_orig.append(y_test_orig_)

            vol_train.append(vol_train_)
            vol_val.append(vol_val_)
            vol_test.append(vol_test_)

        ### End: Loop for Every Asset ######################################################################
        ####################################################################################################

        ####################################################################################################
        ### Start: Stack Datasets & Turn to Torch Tensor ###################################################

        arrays_X        = [X_train, X_val, X_test]
        arrays_not_X    = [y_train, y_val, y_test, y_train_orig, y_val_orig,
                           y_test_orig, vol_train, vol_val, vol_test]

        # Turn arrays from a list of lists of pd.DataFrames to a list of pd.DataFrames
        # (i.e. turn list[pd.DataFrame] to pd.DataFrame by merging across assets)
        X_train, X_val, X_test = list(map(self._to_tensorX, arrays_X))
        y_train, y_val, y_test, y_train_orig, y_val_orig, \
            y_test_orig, vol_train, vol_val, vol_test = list(map(self._to_tensor, arrays_not_X))

        ### End: Stack Datasets & Turn to Torch Tensor #####################################################
        ####################################################################################################

        # Check alignment by time axis
        for i in range(1, len(test_datetimes)):
            assert np.all((test_datetimes[i] - test_datetimes[0]) == pd.Timedelta(0)) 
        assert len(X_test) == len(test_datetimes[0])
        if any(len(x) == 0 for x in [X_train, X_val, X_test]):
            raise ValueError("One of the splits is empty. Check your data and split parameters.")

        ####################################################################################################
        ### Start: Prepare Categorical Data ################################################################

        if self._cat_cols:

            # Create storage
            X_cat = np.zeros((len(train_val_test), self._timesteps, len(self._cat_cols)))
            
            # Fill X
            for i, col in enumerate(self._cat_cols): # for every column
                for j in range(self._timesteps):     # for every timestep
                    X_cat[:, j, i] = train_val_test[col].shift(self._timesteps - j - 1) 
                    # extracts from -timesteps to -0: j=-1 corresponds to .shift(0)

            # Split
            X_cat = torch.from_numpy(X_cat).float()
            train_idx, val_idx, test_idx = train_idx.to_numpy(), val_idx.to_numpy(), test_idx.to_numpy()
            X_train_cat, X_val_cat, X_test_cat = X_cat[train_idx,...], X_cat[val_idx,...], X_cat[test_idx,...]

            X_train_cat = X_train_cat[self._timesteps-1:]

            # NB: categorical variables should be scaled in case of big absolute maximum values
            X_train = torch.cat([X_train, X_train_cat.unsqueeze(-1).expand(-1,-1,-1,len(self._data))], dim=2)
            X_val   = torch.cat([X_val, X_val_cat.unsqueeze(-1).expand(-1,-1,-1,len(self._data))], dim=2)
            X_test  = torch.cat([X_test, X_test_cat.unsqueeze(-1).expand(-1,-1,-1,len(self._data))], dim=2)

            X_cat = X_cat[self._timesteps-1:]

            cat_info = {col_name: len(torch.unique(X_train_cat[..., i])) # TODO: X_train_cat
                        for i, col_name in enumerate(self._cat_cols)}
        else:
            cat_info = {}

        ### End: Prepare Categorical Data ##################################################################
        ####################################################################################################

        ####################################################################################################
        ### Start: Prepare DataLoaders #####################################################################

        # Fix batches sampling order for reproducibility
        g = torch.Generator()   # creates and returns a generator object that manages the state of the algorithm which produces pseudo random numbers
        g.manual_seed(seed)     # sets the seed for generating random numbers and returns a torch.Generator object

        train_loader = DataLoader(TensorDataset(X_train, y_train, y_train_orig, vol_train),
                                  shuffle=True, batch_size=self._batch_size,
                                  worker_init_fn=seed_worker, generator=g,
                                  num_workers=num_workers)

        val_loader = DataLoader(TensorDataset(X_val, y_val, y_val_orig, vol_val),
                                shuffle=False, batch_size=self._batch_size,
                                num_workers=num_workers)

        test_loader = DataLoader(TensorDataset(X_test, y_test, y_test_orig, vol_test),
                                 shuffle=False, batch_size=self._batch_size,
                                 num_workers=num_workers)

        ### End: Prepare DataLoaders #######################################################################
        ####################################################################################################

        return train_loader, val_loader, test_loader, test_datetimes[0], cat_info

        # DataLoader: combines a dataset and a sampler, and provides an iterable over the given dataset
        # TensorDataset(*tensors): Dataset wrapping tensors
        # worker_init_fn (Callable): this will be called on each worker subprocess with the worker id as input, after seeding and before data loading
        # generator (torch.Generator): this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers