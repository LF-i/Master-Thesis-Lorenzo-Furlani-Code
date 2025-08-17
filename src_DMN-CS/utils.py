import torch
import numpy as np
import random
import os
import time
import pickle
from src.models import SLP, Transformer
from src.data import MultivariateTrainValTestSplitter
# from tqdm import tqdm

from empyrical import sharpe_ratio
import matplotlib.pyplot as plt

MODEL_MAPPING = {
    'slp': SLP,
    'trf': Transformer
    }

def _set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}") # TODO


def reg_l1(model, lambda_reg=0.0):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l1_norm = lambda_reg*l1_norm
    return l1_norm

def reg_l2(model, lambda_reg=0.0):
    l2_norm = sum(p.sqrt().sum() for p in model.parameters())
    l2_norm = lambda_reg*l2_norm
    return l2_norm

def sharpe_loss(preds: torch.Tensor, returns: torch.Tensor, volatility: float=0,
                transaction_cost: int=0.0, apply_turnover_reg: bool=False):
    '''
    Loss slightly different than in Lim at al.'s paper.
    Here it's the Sharpe ratio of the portfolio, rather than of the individual assets.
    Args:
        preds: predictions from model (batch_size, n_assets)
        returns:  returns from data (batch_size, n_assets)
        volatility: asset's volatility (batch_size, n_assets)
        transaction_cost: transaction cost to price in
        turnover_reg: True if turnover should be regularized 
    Values:
        sharpe_l: negative annual Sharpe ratio of portfolio returns
    '''
    # # Gross exposure constraint of 1.0
    # preds_sum_per_time = preds.abs().sum(dim=1)
    # norm = torch.clamp_min(preds_sum_per_time, 1.0)
    # preds = preds / norm.unsqueeze(1)

    # Turnover regularization (only if enabled)
    # if apply_turnover_reg:
    #     target_vol = 0.15
    #     T = target_vol * torch.abs(
    #         torch.diff(
    #             preds / (volatility * 252**0.5 + 1e-9),
    #             dim=0,
    #             prepend=torch.zeros((1, preds.shape[1]), device=preds.device, dtype=preds.dtype)
    #         )
    #     )
    #     R = preds * returns - 1e-4 * transaction_cost * T
    # else:
    #     R = preds * returns
    R = preds * returns

    # Average of asset-level Sharpe ratios
    mu_assets       = R.mean(dim=0)
    sigma_assets    = R.std(dim=0, unbiased=True)
    sharpe_assets   = -252**0.5 * mu_assets / (sigma_assets + 1e-9)
    sharpe_loss     = sharpe_assets.mean()

    # Portfolio's Sharpe ratio
    # R_pf        = R.mean(dim=1)
    # R_mean      = R_pf.mean(dim=0)
    # R_std       = R_pf.std(dim=0, unbiased=True)
    # sharpe_loss = -252**0.5 * R_mean / (R_std + 1e-9)

    return sharpe_loss


## Simplify training step
def train(model, train_loader, optimizer, device: str, 
          max_norm: bool=None, apply_l1_reg: bool=False, l1_reg_lambda: float=None,
          apply_turnover_reg: bool=False, transaction_cost: int=0.0):

    train_loss    = 0.0
    train_l1_loss = 0.0
    model.train()

    for batch_data in train_loader: # earlier: tqdm(train_loader)
    
        batch_x, batch_y, _, batch_vol = (x.to(device) for x in batch_data)

        output = model(batch_x)

        l = sharpe_loss(output, batch_y.flatten(), batch_vol, transaction_cost, apply_turnover_reg)
        train_loss += l.item()

        if apply_l1_reg:
            l_l1 = reg_l1(model, l1_reg_lambda)
            train_l1_loss += l_l1.item()
            l += l_l1

        optimizer.zero_grad()
        l.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    
    return train_loss / len(train_loader), train_l1_loss / len(train_loader)


def evaluate(model, val_loader, device: str,
             apply_l1_reg: bool=False, l1_reg_lambda: float=0.0, 
             apply_turnover_reg: bool=False, transaction_cost: int=0.0):

    val_l1_loss         = 0.0
    preds               = []
    returns             = []
    vols                = []
    model.eval()

    # Evaluate model performance on validation dataset
    with torch.no_grad():
        for batch_data in val_loader:
            batch_x, batch_y, _, batch_vol = (x.to(device) for x in batch_data)
            output = model(batch_x)

            returns.append(batch_y.detach().cpu())
            preds.append(output.detach().cpu())
            vols.append(batch_vol.detach().cpu())

    returns = torch.cat(returns, dim=0)
    preds   = torch.cat(preds, dim=0)
    vols    = torch.cat(vols, dim=0)

    l = sharpe_loss(preds, returns.flatten(), vols, transaction_cost, apply_turnover_reg)
    val_loss = l.item()

    if apply_l1_reg:
        l_l1 = reg_l1(model, l1_reg_lambda)
        val_l1_loss = l_l1.item()

    # Convert to numpy arrays for stats
    returns, preds, vols = returns.numpy(), preds.numpy(), vols.numpy()

    stats   = [returns, preds, vols]
    losses  = [val_loss, val_l1_loss]

    return stats, losses


def output_stats(model, loader, device, transaction_cost, apply_turnover_reg
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

    # Pre-calculate total size
    total_size = len(loader.dataset)

    # Pre-allocate tensors
    preds           = []
    returns         = []
    returns_orig    = []
    vols            = []

    # Calculate model predictions on dataset
    idx = 0
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            batch_x, batch_y, batch_y_orig, batch_vol = (x.to(device) for x in batch_data)
            curr_batch_size = batch_x.size(0)

            output = model(batch_x)

            preds.append(output.detach().cpu())
            returns.append(batch_y.detach().cpu())
            returns_orig.append(batch_y_orig.detach().cpu())
            vols.append(batch_vol.detach().cpu())

            idx += curr_batch_size

    preds   = torch.cat(preds, dim=0)
    returns = torch.cat(returns, dim=0)
    returns_orig = torch.cat(returns_orig, dim=0)
    vols    = torch.cat(vols, dim=0)

    loss = sharpe_loss(preds, returns.flatten(), vols, transaction_cost, apply_turnover_reg)

    return preds.numpy(), returns.numpy(), returns_orig.numpy(), vols.numpy(), loss.item()


def grid_search_iter(splitter, start, test_delta, device, 
                     iteration_params, fixed_params, model_type, 
                     params_c, target_vol, settings, iter_path):

    checkpoint_path = os.path.join('weights', iter_path + '_' + str(start.year))
    
    save_model  = settings['save_model']
    plot        = settings['plot']
    print_      = settings['print_']

    train_loader, val_loader, test_loader, test_dt, cat_info = splitter.split(
        start, test_delta, iteration_params['seed']
        )

    if len(train_loader) == 0 or len(val_loader) == 0 or len(test_loader) == 0:
        return None

    ## Create model
    _set_seed(iteration_params['seed'])
    batch_data = next(iter(train_loader)); batch_x, batch_y, _, _ = batch_data; 

    # n_features, n_assets, n_shared = params
    # assert batch_x.shape[-1] == n_features*n_assets + n_shared + len(cat_info)

    iteration_params['n_features']  = batch_x.shape[-2] - len(cat_info)
    iteration_params['n_assets']    = batch_x.shape[-1]
    iteration_params['dim_out']     = 1
    iteration_params['cat_info']    = cat_info

    model   = MODEL_MAPPING[model_type](timesteps=fixed_params['timesteps'], **iteration_params).to(device)
    opt     = torch.optim.Adam(params=model.parameters(), lr=iteration_params['lr'], weight_decay=iteration_params['l2_reg_weight'])
    torch.save(model.state_dict(), checkpoint_path)

    ## Create storage
    results                 = {}    
    train_losses            = []
    train_l1_losses         = []
    val_losses              = []
    val_SRs_per_epoch       = []
    best_val_loss           = np.inf
    counter                 = 0

    ### START: EPOCHS

    for e in range(fixed_params['n_epochs']):
    # for e in tqdm(range(fixed_params['n_epochs'])): # for every epoch # TODO

        train_loss, train_l1_loss = train(
            model=model, train_loader=train_loader, optimizer=opt, 
            device=device, max_norm=iteration_params['max_norm'], apply_l1_reg=fixed_params['apply_l1_reg'], 
            l1_reg_lambda=iteration_params['l1_reg_weight'], apply_turnover_reg=params_c['apply_turnover_reg'], 
            transaction_cost=params_c['transaction_cost'],
            )

        stats, losses = evaluate(
            model=model, val_loader=val_loader, device=device,
            apply_l1_reg=fixed_params['apply_l1_reg'], l1_reg_lambda=iteration_params['l1_reg_weight'],
            apply_turnover_reg=params_c['apply_turnover_reg'], transaction_cost=params_c['transaction_cost']
            )
        val_returns, val_preds, val_vols    = stats
        val_loss, val_l1_loss   = losses

        # If the current validation loss is smaller, save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            save_attempts = 0

            while save_attempts < 5:
                try:
                    torch.save(model.state_dict(), checkpoint_path)
                    break
                except (PermissionError, RuntimeError) as e:
                    save_attempts += 1
                    time.sleep(0.5)
                    if save_attempts == 5:
                        raise e
        else:
            counter += 1

        # If metric value didn't improve for several epochs, stop training
        if counter > fixed_params['patience']:
            break

        if plot or print_:
            # Validation turnover
            val_vols = val_vols * 252**0.5
            val_preds = val_preds.resize((int(val_preds.shape[0]/2),2))
            T = target_vol*np.abs(np.diff(val_preds/(val_vols+1e-9), prepend=0.0, axis=0))
            val_SRs_per_c = {}
            for c in params_c['basis_points']:
                captured = val_returns*val_preds - 1e-4*c*T 
                R = np.mean(captured, axis=1)
                val_SRs_per_c[c] = sharpe_ratio(R)

            val_SRs_per_epoch.append(val_SRs_per_c)

        # Aggregate losses
        train_losses.append(train_loss)
        train_l1_losses.append(train_l1_loss)
        val_losses.append(val_loss)

        if print_:
            print('Epoch:', e)
            print('Train loss:', round(train_losses[-1], 3))
            print('Val loss:', round(val_losses[-1], 3))
            if fixed_params['apply_l1_reg']:
                print('L1 loss:', round(train_l1_losses[-1], 5))
            print('Val Sharpe ratio')
            for key in val_SRs_per_c.keys():
                print('c:', key, 'SR:', round(val_SRs_per_c[key], 3))
            print('Nr. of epochs without improvement in val loss:', counter if e > 1 else 0)
            print()

    ### END: EPOCHS

    if plot:
        # Prepare timelines of valuation Sharpe ratios
        val_SRs_timeline = np.zeros((len(params_c['basis_points'] ), len(val_SRs_per_epoch)))
        for i, val_SRs_per_c in enumerate(val_SRs_per_epoch):
            val_SRs_timeline[:,i] = np.array(list(val_SRs_per_c.values()))

        print('Validatin up to date:', start)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # 1 row, 3 columns

        axes[0].set_title('Validation Loss Evolution')
        axes[0].plot(val_losses, label='validation', marker='o')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].legend()

        if fixed_params['apply_l1_reg']:
            axes[1].set_title('L1 Regularization Train Loss Evolution')
            axes[1].plot(train_l1_losses, label='train', marker='o')
            axes[1].set_ylabel('L1 Loss')
            axes[1].set_xlabel('Epochs')
            axes[1].legend()
        else:
            axes[1].axis('off')

        axes[2].set_title('Valuation Sharpe Ratios Timeline')
        for c, timeline in zip(val_SRs_per_c.keys(), val_SRs_timeline):
            axes[2].plot(timeline, label=f'C: {c}', marker='o')
        axes[2].set_ylabel('Sharpe ratio')
        axes[2].set_xlabel('Epochs')
        axes[2].legend()

        plt.tight_layout()
        plt.show()
        plt.close('all')

    ### END: GRID SEARCH

    # Load best checkpoint in terms of validation loss
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    os.remove(checkpoint_path)

    val_preds, val_returns, val_returns_orig, val_vols, val_loss = output_stats(
        model=model, loader=val_loader, device=device,
        apply_turnover_reg=params_c['apply_turnover_reg'], transaction_cost=params_c['transaction_cost']
        )
    test_preds, test_returns, test_returns_orig, test_vols, test_loss = output_stats(
        model=model, loader=test_loader, device=device,
        apply_turnover_reg=params_c['apply_turnover_reg'], transaction_cost=params_c['transaction_cost']
        )

    results['val']                  = {}
    results['test']                 = {}
    results['test_dt']              = test_dt
    results['best_loss']            = best_val_loss
    results['val']['preds']         = val_preds
    results['val']['returns']       = val_returns
    results['val']['returns_orig']  = val_returns_orig
    results['val']['vols']          = val_vols
    results['val']['loss']          = val_loss
    results['test']['preds']        = test_preds
    results['test']['returns']      = test_returns
    results['test']['returns_orig'] = test_returns_orig
    results['test']['vols']         = test_vols
    results['test']['loss']         = test_loss
    if save_model:
        results['model']            = model

    return results


def train_model_slp(hyperparams, date_range, model_type, params_c, fixed_params,
                    features, cols_to_use, datetime_cols, test_delta, device,
                    target_vol, settings):

    if not os.path.exists('weights'):
        os.mkdir('weights')
    if not os.path.exists('results'):
        os.mkdir('results')

    iteration_params = {
        'batch_size': hyperparams['batch_size'],
        'lr': hyperparams['lr'],
        'max_norm': hyperparams['max_norm'],
        'l1_reg_weight': hyperparams['l1_reg_weight'],
        'dropout': None,
        'n_heads': None,
        'n_enc_layers': None,
        'n_dec_layers': None,
        'n_layers': None,
        'd_model': None,
        'seed': hyperparams['seed']
    }

    # Construct the file path for the results file
    iter_path = os.path.join(
        '{}_seed{}_c{}_lr{}'.format(model_type, iteration_params['seed'], params_c['transaction_cost'], iteration_params['lr'])
        + f"_batch{iteration_params['batch_size']}"
        + f"_max{iteration_params['max_norm']}"
        + (f"_l1{iteration_params['l1_reg_weight']}")
        )
    results_path = os.path.join('results', iter_path) + '.pickle'
    
    # Check if the results file already exists
    if os.path.exists(results_path):
        return

    splitter = MultivariateTrainValTestSplitter(
        data=features, cols=cols_to_use, cat_cols=datetime_cols,
        target_col='target_returns', orig_returns_col='target_returns_nonscaled', 
        vol_col='daily_vol', scaling=fixed_params['scaling'],
        timesteps=fixed_params['timesteps'], batch_size=iteration_params['batch_size']
        )

    results = {}

    for start in date_range:

        results[start] = grid_search_iter(splitter=splitter, start=start, test_delta=test_delta,
                                        device=device, iteration_params=iteration_params,
                                        fixed_params=fixed_params, model_type=model_type,
                                        params_c=params_c, target_vol=target_vol,
                                        settings=settings, iter_path=iter_path)

    # Dump results of experiment
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    return


def train_model_trf(hyperparams, date_range, model_type, params_c, fixed_params,
                    features, cols_to_use, datetime_cols, shared_cols, test_delta, device,
                    target_vol, settings):

    if not os.path.exists('weights'):
        os.mkdir('weights')
    if not os.path.exists('results'):
        os.mkdir('results')

    iteration_params = {
        'batch_size': hyperparams['batch_size'],
        'lr': hyperparams['lr'],
        'max_norm': hyperparams['max_norm'],
        'dropout': hyperparams['dropout'],
        'n_heads': hyperparams['n_heads'],
        'n_enc_layers': hyperparams['n_layers'],
        'n_dec_layers': hyperparams['n_layers'],
        'n_layers': hyperparams['n_layers'],
        'd_model': hyperparams['d_model'],
        'multiplier': hyperparams['multiplier'],
        'l1_reg_weight': hyperparams['l1_reg_weight'] if hyperparams['l1_reg_weight'] is not None else 0.0,
        'l2_reg_weight': hyperparams['l2_reg_weight'] if hyperparams['l2_reg_weight'] is not None else 0.0,
        'seed': hyperparams['seed']
    }

    # Construct the file path for the results file
    iter_path = os.path.join(
        '{}_s{}_c{}_lr{}'.format(model_type, iteration_params['seed'], params_c['transaction_cost'], iteration_params['lr'])
        + f"_b{iteration_params['batch_size']}"
        + (f"_drop{iteration_params['dropout']}" if iteration_params['dropout'] is not None else "")
        + f"_max{iteration_params['max_norm']}"
        + (f"_h{iteration_params['n_heads']}" if iteration_params['n_heads'] is not None else "")
        + (f"_M{iteration_params['n_layers']}" if iteration_params['n_layers'] is not None else "")
        + (f"_dm{iteration_params['d_model']}" if iteration_params['d_model'] is not None else "")
        + (f"_m{iteration_params['multiplier']}" if iteration_params['multiplier'] is not None else "")
        + (f"_l1{iteration_params['l2_reg_weight']}" if iteration_params['l2_reg_weight']!=0.0 else "")
        + (f"_l2{iteration_params['l2_reg_weight']}" if iteration_params['l2_reg_weight']!=0.0 else "")
        )
    results_path = os.path.join('results', iter_path) + '.pickle'

    # Check if the results file already exists
    if os.path.exists(results_path):
        return

    splitter = MultivariateTrainValTestSplitter(
        data=features, cols=cols_to_use, cat_cols=datetime_cols, shared_cols=shared_cols,
        target_col='target_returns', orig_returns_col='target_returns_nonscaled',
        vol_col='daily_vol', scaling=fixed_params['scaling'],
        timesteps=fixed_params['timesteps'], batch_size=iteration_params['batch_size']
        )

    results = {}

    for start in date_range:

        results[start] = grid_search_iter(splitter=splitter, start=start, test_delta=test_delta,
                                        device=device, iteration_params=iteration_params,
                                        fixed_params=fixed_params, model_type=model_type,
                                        params_c=params_c, target_vol=target_vol,
                                        settings=settings, iter_path=iter_path)

    # Dump results of experiment
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    return


