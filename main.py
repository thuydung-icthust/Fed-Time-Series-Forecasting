import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import wandb

import sys
import os
import config

from argparse import Namespace

from pathlib import Path
from fl_trainer import fit, inference
parent = Path(os.path.abspath("")).resolve().parents[0]

if parent not in sys.path:
    sys.path.insert(0, str(parent))

from helpers import *

def main(args):
    wandb_instance = wandb.init(project=config.WANDB_PROJECT,
                                entity=config.WANDB_ENTITY,
                                name=args.instance,
                                config=args)
    seed_all(args)
    # Get training data
    X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers = make_preprocessing(args)
    X_train, X_val, y_train, y_val, client_X_train, client_X_val, client_y_train, client_y_val, exogenous_data_train, exogenous_data_val = make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers, args)
    X_val_sub, Y_val_sub = get_subset_val(client_X_val, client_y_val, 0.1)
    subval_dataloader =  to_torch_dataset(X_val_sub, Y_val_sub,
                                        num_lags=args.num_lags,
                                        num_features = len(X_val_sub[0][0]),
                                        exogenous_data=None,
                                        indices=[8, 3, 1, 10, 9],
                                        batch_size=args.batch_size,
                                        shuffle=False)
    # print dict keys for all clients
    print(f"Clients participating training are: {client_X_val.keys()}")
    
    input_dim, exogenous_dim = get_input_dims(X_train, exogenous_data_train, args)
    model = get_model(model=args.model_name,
                  input_dim=input_dim,
                  out_dim=y_train.shape[1],
                  lags=args.num_lags,
                  exogenous_dim=exogenous_dim,
                  seed=args.seed)
    # federated local params
    local_train_params = {"epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr,
                        "criterion": args.criterion, "early_stopping": args.local_early_stopping,
                        "patience": args.local_patience, "device": args.device
                        }
    hier_dict = get_hier_dict(client_X_train.keys())
    
    global_model, history = fit(
                                model,
                                client_X_train,
                                client_y_train, 
                                client_X_val, 
                                client_y_val, 
                                local_train_params=local_train_params,
                                args = args, 
                                wandb_ins = wandb_instance,
                                subval_dataloader = subval_dataloader,
                                hier_dict = hier_dict)
    
    # validation_dict = inference(
    #                 global_model,
    #                 client_X_train, 
    #                 client_y_train,
    #                 client_X_val, 
    #                 client_y_val,
    #                 exogenous_data_train, 
    #                 exogenous_data_val,
    #                 y_scalers,
    #                 args)
    # for key_ in validation_dict.keys():
    #     wandb.log({f"{key_}/val_mse": validation_dict[key_]['val_mse'],
    #                f"{key_}/val_rmse": validation_dict[key_]['val_rmse'],
    #                f"{key_}/val_mae": validation_dict[key_]['val_mae'],
    #                f"{key_}/val_r2": validation_dict[key_]['val_r2'],
    #                f"{key_}/val_nrmse": validation_dict[key_]['val_nrmse'],
    #                f"{key_}/flr": flr
    #                })
        
        #     for key_ in validation_dict.keys():
        # wandb.log({f"{key_}/":             'val_mse': val_mse,
        #     'val_rmse': val_rmse,
        #     'val_mae': val_mae,
        #     'val_r2': val_r2,
        #     'val_nrmse': val_nrmse} for val_descriptor)

def read_config_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        config = json.load(f)
    return config
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file")
    parser.add_argument("--config", type=str, help="Path to configuration file", required=True)
    args = parser.parse_args()
    config_from_json = read_config_from_json(args.config)

    default_params = {
        "data_path": 'dataset/full_dataset.csv',
        "test_size": 0.2,
        "targets": ['rnti_count', 'rb_down', 'rb_up', 'down', 'up'],
        "num_lags": 10,
        "identifier": 'District',
        "nan_constant": 0,
        "x_scaler": 'minmax',
        "y_scaler": 'minmax',
        "outlier_detection": True,
        "defense_on": 0,
        "criterion": 'mse',
        "fl_rounds": 20,
        "fraction": 1.,
        "aggregation": "fedavg",
        "epochs": 3,
        "lr": 0.001,
        "optimizer": 'adam',
        "batch_size": 128,
        "local_early_stopping": False,
        "local_patience": 50,
        "max_grad_norm": 0.0,
        "reg1": 0.0,
        "reg2": 0.0,
        "model_name": "rnn",
        "num_subset": 5,
        "device": "cuda",
        "instance": "baseline-fedavg",
        "cuda": True,
        "seed": 0,
        "assign_stats": None,
        "use_time_features": False
    }
    
    # args_default = Namespace(
    #     data_path='dataset/full_dataset.csv', # dataset
    #     test_size=0.2, # validation size 
    #     targets=['rnti_count', 'rb_down', 'rb_up', 'down', 'up'], # the target columns
    #     num_lags=10, # the number of past observations to feed as input
    #     identifier='District', # the column name that identifies a bs
    #     nan_constant=0, # the constant to transform nan values
    #     x_scaler='minmax', # x_scaler
    #     y_scaler='minmax', # y_scaler
    #     outlier_detection=True, # whether to perform flooring and capping
    #     criterion='mse', # optimization criterion, mse or l1
    #     fl_rounds=20, # the number of federated rounds
    #     fraction=1., # the percentage of available client to consider for random selection
    #     aggregation="fedavg", # federated aggregation algorithm
    #     epochs=3, # the number of maximum local epochs
    #     lr=0.001, # learning rate
    #     optimizer='adam', # the optimizer, it can be sgd or adam
    #     batch_size=128, # the batch size to use
    #     local_early_stopping=False, # whether to use early stopping
    #     local_patience=50, # patience value for the early stopping parameter (if specified)
    #     max_grad_norm=0.0, # whether to clip grad norm
    #     reg1=0.0, # l1 regularization
    #     reg2=0.0, # l2 regularization
    #     model_name="rnn",
    #     num_subset=5,
    #     has_attack=0,
    #     device="cuda",
    #     instance="baseline-fedavg",
    #     cuda=True, # whether to use gpu
    #     seed=0, # reproducibility
    #     assign_stats=None, # whether to use statistics as exogenous data, ["mean", "median", "std", "variance", "kurtosis", "skew"]
    #     use_time_features=False # whether to use datetime features
    # )
    # Merge the default parameters with the loaded JSON configuration
    args = {**default_params, **config_from_json}
    args = Namespace(**args)
    print(args)
    if args.outlier_detection is not None:
        outlier_columns = ['rb_down', 'rb_up', 'down', 'up']
        outlier_kwargs = {"ElBorn": (10, 90), "LesCorts": (10, 90), "PobleSec": (5, 95)}
        args.outlier_columns = outlier_columns
        args.outlier_kwargs = outlier_kwargs
    main(args)



