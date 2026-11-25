# PADformer source; data and trained artifacts are not included.

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ModuleNotFoundError:
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None

def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M')
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    logfile = os.path.join(root, 'run.log')
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger

def init_seed(seed, strict=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = not strict
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = not strict
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest' if strict else 'high')
    torch.use_deterministic_algorithms(strict, warn_only=False)
    torch.backends.cudnn.enabled = True

def All_Metrics(pred, true):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
    if isinstance(true, torch.Tensor):
        true = true.cpu().detach().numpy()
    mask = true > 0
    if np.sum(mask) == 0:
        return (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    true_valid = true[mask]
    pred_valid = pred[mask]
    if mean_absolute_error is None:
        mae = np.mean(np.abs(true_valid - pred_valid))
        rmse = np.sqrt(np.mean((true_valid - pred_valid) ** 2))
        ss_res = np.sum((true_valid - pred_valid) ** 2)
        ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
        r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    else:
        mae = mean_absolute_error(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        r2 = r2_score(true_valid, pred_valid)
    mape = np.mean(np.abs((true_valid - pred_valid) / true_valid))
    return (torch.tensor(mae), torch.tensor(rmse), torch.tensor(mape), torch.tensor(r2))
