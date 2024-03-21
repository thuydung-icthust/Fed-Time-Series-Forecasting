"""
Implements the server and the federated process.
"""

from collections import OrderedDict
import copy
import sys

from pathlib import Path
from termcolor import colored

from sklearn import preprocessing

from helpers import get_hier_dict, norm_clipping
min_max_scaler = preprocessing.MinMaxScaler()

import torch

from ml.utils.train_utils import get_preds, load_model_weight, test, vectorize_net
from verifier import decompose_ts, sat_verifier, sat_verifier_all_feats, trend_verifier

parent = Path(__file__).resolve().parents[3]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import time
from logging import DEBUG, INFO
from typing import Optional, Callable, List, Tuple, Dict, Union

import numpy as np
from torch.utils.data import DataLoader

from ml.fl.server.client_proxy import ClientProxy
from ml.fl.server.client_manager import ClientManager, SimpleClientManager

from ml.utils.logger import log
from ml.fl.history.history import History

from ml.fl.server.aggregation.aggregator import Aggregator
from ml.fl.defaults import weighted_loss_avg, weighted_metrics_avg


class Server:
    def __init__(self,
                 client_proxies: List[ClientProxy],
                 client_manager: Optional[ClientManager] = None,
                 aggregation: Optional[str] = None,
                 aggregation_params: Optional[Dict[str, Union[str, int, float, bool]]] = None,
                 weighted_loss_fn: Optional[Callable] = None,
                 weighted_metrics_fn: Optional[Callable] = None,
                 val_loader: Optional[DataLoader] = None,
                 local_params_fn: Optional[Callable] = None,
                 global_val_loaders: Optional[DataLoader] = None,
                 model=None,
                 subval_dataloader=None,
                 defense_on=False,
                 hier_dict={}):
        
        self.model_in = model
        self.global_model = None
        self.best_model = None
        self.best_loss, self.best_epoch = np.inf, -1
        self.defense_on = defense_on

        self.client_proxies = client_proxies
        self._initialize_client_manager(client_manager)  # initialize the client manager

        self.weighted_loss = weighted_loss_fn if weighted_loss_fn is not None else weighted_loss_avg
        self.weighted_metrics = weighted_metrics_fn if weighted_metrics_fn is not None else weighted_metrics_avg
        self.global_val_loaders = global_val_loaders
        self.val_subset = subval_dataloader
        self.hier_dict = hier_dict
        
        if aggregation is None:
            aggregation = "fedavg"
        self.aggregator = Aggregator(aggregation_alg=aggregation, params=aggregation_params)
        log(INFO, f"Aggregation algorithm: {repr(self.aggregator)}")
        
        # TODO: change it later
        self.fedprox_mu = 0.0
        if aggregation == "fedprox":
            self.fedprox_mu = self.aggregator.mu
            local_params_fn = lambda fl_round: {"fedprox_mu": self.fedprox_mu}
        self.val_loader = val_loader
        self.local_params_fn = local_params_fn

    def _initialize_client_manager(self, client_manager) -> None:
        """Initialize client manager"""
        log(INFO, "Initializing client manager...")
        if client_manager is None:
            client_manager: ClientManager = SimpleClientManager()
            self.client_manager = client_manager
        else:
            self.client_manager = client_manager

        log(INFO, "Registering clients...")
        for client_proxy in self.client_proxies:  # register clients
            self.client_manager.register(client_proxy)
        log(INFO, "Client manager initialized!")

    def fit(self,
            num_rounds: int,
            fraction: float,
            fraction_args: Optional[Callable] = None,
            use_carbontracker: bool = True,
            wandb_ins = None,
            malicious_idxs = []) -> Tuple[List[np.ndarray], History]:
        """Run federated rounds for num_rounds rounds."""

        history = History()

        self.evaluate_round(fl_round=0, history=history)

        log(INFO, "Starting FL rounds")
        cb_tracker = None
        if use_carbontracker:
            try:
                from carbontracker.tracker import CarbonTracker
                cb_tracker = CarbonTracker(epochs=num_rounds, components="all", verbose=1)
            except ImportError:
                pass

        start_time = time.time()

        for fl_round in range(1, num_rounds + 1):
            if use_carbontracker and cb_tracker is not None:
                cb_tracker.epoch_start()
            # train and replace the previous global model
            self.fit_round(fl_round=fl_round,
                           fraction=fraction,
                           fraction_args=fraction_args,
                           history=history, 
                           malicious_idxs=malicious_idxs)
            if use_carbontracker and cb_tracker is not None:
                cb_tracker.epoch_end()
            # evaluate global model
            # test_metrics = self.evaluate_round(fl_round=fl_round,
            #                     history=history)
            random_net = copy.deepcopy(self.client_manager.sample(0.)[0])
            test_metrics = self.random_evaluate(self.global_val_loaders, self.global_model, net=random_net)
            for key_ in test_metrics.keys():
                wandb_ins.log({f"{key_}/val_mse": test_metrics[key_]['MSE'],
                           f"{key_}/val_rmse": test_metrics[key_]['RMSE'],
                           f"{key_}/val_mae": test_metrics[key_]['MAE'],
                           f"{key_}/val_r2": test_metrics[key_]['R^2'],
                           f"{key_}/val_nrmse": test_metrics[key_]['NRMSE'],
                           f"{key_}/flr": fl_round
                           })
            
        end_time = time.time()
        # log(INFO, history)
        log(INFO, f"Time passed: {end_time - start_time} seconds.")
        log(INFO, f"Best global model found on fl_round={self.best_epoch} with loss={self.best_loss}")

        return self.best_model, history

    def fit_round(self, fl_round: int,
                  fraction: float,
                  fraction_args: Optional[Callable],
                  history: History, 
                  malicious_idxs = []) -> None:
        """Perform a federated round, i.e.,
            1) Select a fraction of available clients.
            2) Instruct selected clients to execute local training.
            3) Receive updated parameters from clients and their corresponding evaluation
            4) Aggregate the local learned weights.
        """
        # Inform clients for local parameters change if any
        if self.local_params_fn:
            for client_proxy in self.client_proxies:
                client_proxy.set_train_parameters(self.local_params_fn(fl_round), verbose=True)

        # STEP 1: Select a fraction of available clients
        selected_clients = self.sample_clients(fl_round, fraction, fraction_args)
        client_keys = [client.cid for client in selected_clients]
        self.hier_dict = get_hier_dict(client_keys)
        # print(f"selected_clients: {client_keys}")
        # STEPS 2-3: Perform local training and receive updated parameters
        num_train_examples: List[int] = []
        num_test_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        test_losses: Dict[str, float] = dict()
        all_train_metrics: Dict[str, Dict[str, float]] = dict()
        all_test_metrics: Dict[str, Dict[str, float]] = dict()
        results: List[Tuple[List[np.ndarray], int]] = []
        model_vec_list = []
        
        for client in selected_clients:
            res = self.fit_client(fl_round, client)
            model_params, model_vec, num_train, train_loss, train_metrics, num_test, test_loss, test_metrics = res
            num_train_examples.append(num_train)
            num_test_examples.append(num_test)
            train_losses[client.cid] = train_loss
            test_losses[client.cid] = test_loss
            all_train_metrics[client.cid] = train_metrics
            all_test_metrics[client.cid] = test_metrics
            results.append((model_params, num_train))
            model_vec_list.append(model_vec)

        history.add_local_train_loss(train_losses, fl_round)
        history.add_local_train_metrics(all_train_metrics, fl_round)
        history.add_local_test_loss(test_losses, fl_round)
        history.add_local_test_metrics(all_test_metrics, fl_round)

        # Conduct defense before aggregation
        verifier_w = []
        # if self.defense_on:
        #     print(colored("----------*----------\n CONDUCTING DEFENSE:", "green"))
        #     verifier_w = self.conduct_defense(model_vec_list, fl_round, malicious_idxs).squeeze()
        #     # print(f"Full features verifiers: {verifier_w}")
        #     # import IPython
        #     # IPython.embed()
        #     v_scaled = np.mean(verifier_w, axis=0)
        #     v_scaled = min_max_scaler.fit_transform(v_scaled.reshape(-1, 1))
            
            
        #     # v_scaled = min_max_scaler.fit_transform(verifier_w.T).T
        #     # v_scaled = np.mean(v_scaled, axis=0)
            
            
        #     # v_scaled = min_max_scaler.fit_transform(v_scaled.reshape(-1, 1))
        #     # import IPython
        #     # IPython.embed()
            
        #     print(colored(f"Malicious ids: {malicious_idxs}", "green"))
        #     good_idxes = [id_v for id_v, v in enumerate(v_scaled) if v >= np.median(v_scaled)]
        #     clipping_idxs = [id_v for id_v, v in enumerate(v_scaled) if v < np.median(v_scaled)]
        #     print(colored(f"Coffs ids: {v_scaled}, \tMedian is: {np.median(v_scaled)}\nGood idxs: {good_idxes}", "green"))
        #     print(colored("----------*----------", "green"))
        #     v_scaled = [0.0 if i in good_idxes else val for i, val in enumerate(v_scaled)]
        #     print(colored(f"Malicious ids: {clipping_idxs}", "red"))
            
        #     prev_global_model = self.set_parameters(self.model_in, self.global_model).to("cuda")
            
        #     vec_prev_model = vectorize_net(prev_global_model)
            
        #     results_ = norm_clipping(model_vec_list, vec_prev_model, clipping_idxs)
        #     results_ = [model_vec.cpu().detach().numpy() for model_vec in results_]
        #     reconstructed_freq = [num_dpt for (_, num_dpt) in results]
        #     reconstructed_freq = [freq/sum(reconstructed_freq) for freq in reconstructed_freq]
        #     aggregated_grad = np.average(np.array(results_), weights=reconstructed_freq, axis=0).astype(np.float32)
        #     aggregated_model = self.model_in # slicing which doesn't really matter
        #     load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to("cuda"))
        #     results = [([val.cpu().numpy() for _, val in aggregated_model.state_dict().items()], 1.0)]
        #     #TODO: change this stupid snippet later
        
        if self.defense_on:
            print(colored("----------*----------\n CONDUCTING DEFENSE:", "green"))
            malicious_idxs = self.conduct_hier_defense(model_vec_list, fl_round, malicious_idxs)
            print(colored(f"Malicious IDs: {malicious_idxs}", "red"))
            prev_global_model = self.set_parameters(self.model_in, self.global_model).to("cuda")
            vec_prev_model = vectorize_net(prev_global_model)
            
            results_ = norm_clipping(model_vec_list, vec_prev_model, malicious_idxs)
            results_ = [model_vec.cpu().detach().numpy() for model_vec in results_]
            reconstructed_freq = [num_dpt for (_, num_dpt) in results]
            reconstructed_freq = [freq/sum(reconstructed_freq) for freq in reconstructed_freq]
            aggregated_grad = np.average(np.array(results_), weights=reconstructed_freq, axis=0).astype(np.float32)
            aggregated_model = self.model_in # slicing which doesn't really matter
            load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to("cuda"))
            results = [([val.cpu().numpy() for _, val in aggregated_model.state_dict().items()], 1.0)]
            print(colored("----------*----------", "green"))
        else:
            v_scaled = []
        # STEP 4: Aggregate local models
        
        v_scaled = []
        
        self.global_model = self.aggregate_models(fl_round, results, v_scaled)
        
        if self.best_model is None:
            self.best_model = copy.deepcopy(self.global_model)

    def sample_clients(self, fl_round: int, fraction: float,
                    fraction_args: Optional[Callable] = None) -> List[ClientProxy]:
        """Sample available clients."""
        if fraction_args is not None:
            fraction: float = fraction_args(fl_round)

        selected_clients: List[ClientProxy] = self.client_manager.sample(fraction)
        #log(DEBUG, f"[Global round {fl_round}] Sampled {len(selected_clients)} clients "
        #           f"(out of {self.client_manager.num_available(verbose=False)})")

        return selected_clients

    def conduct_defense(self, local_weights, fl_round=0, malicious_idxs=[]):
        # TODO: SAT defense function
        # First, convert all weights to model for inference

        net_list = []
        for weight in local_weights:
            cp_mod = copy.deepcopy(self.model_in)
            load_model_weight(cp_mod, weight)
            net_list.append(cp_mod)
        preds = []
        y_true = []
        for net in net_list:
            y_pred, y_true = get_preds(net, self.val_subset, "cuda")
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            preds.append(y_pred)
        # verifier_w = sat_verifier(preds, y_true, 0)
        verifier_w = trend_verifier(preds, malicious_idxs, fl_round, y_true)
        return verifier_w      
    
    def conduct_hier_defense(self, local_weights, fl_round=0, malicious_idxs=[]):
        # This function conducts hierarchical FL defense, each station will verify their children.
        final_clipping_idxs = []
        for idx, station_name in enumerate(self.hier_dict.keys()):
            children_idxs = self.hier_dict[station_name]
            children_models = [local_weights[id_] for id_ in children_idxs]
            verifier_w = self.conduct_defense(children_models, fl_round, malicious_idxs)
            v_scaled = np.mean(verifier_w, axis=0)
            v_scaled = min_max_scaler.fit_transform(v_scaled.reshape(-1, 1))
            clipping_idxs = [id_v for id_v, v in enumerate(v_scaled) if v < np.median(v_scaled)]
            # import IPython
            # IPython.embed()
            children_idxs = np.asarray(children_idxs)
            
            # for id_v in clipping_idxs:
            final_clipping_idxs.extend(children_idxs[clipping_idxs].tolist())

            # global_clipping_idxs.extend(children_idxs[id_v] for id_v in clipping_idxs)
            # clipping_idxs = [children_idxs[id_v] for id_, id_v in enumerate(clipping_idxs)]
            # final_clipping_idxs.extend(clipping_idxs)
        
        # import IPython
        # IPython.embed()
        return final_clipping_idxs
    
    def fit_client(self,
                   fl_round: int,
                   client: ClientProxy) -> Tuple[
        List[np.ndarray], int, float, Dict[str, float], int, float, Dict[str, float]]:
        """Perform local training."""
        #log(INFO, f"[Global round {fl_round}] Fitting client {client.cid}")
        if fl_round == 1:
            fit_res = client.fit(None)
        else:
            fit_res = client.fit(model=self.global_model)

        return fit_res

    def aggregate_models(self, 
                         fl_round: int, 
                         results: List[Tuple[List[np.ndarray], int]],
                         w = [],
                         ) -> List[np.ndarray]:
        log(INFO, f"[Global round {fl_round}] Aggregating local models...")
        if not len(w):
            aggregated_params = self.aggregator.aggregate(results, self.global_model, [])
        else:
            aggregated_params = self.aggregator.aggregate(results, self.global_model, coffs=w)
        return aggregated_params

    def evaluate_round(self, fl_round: int, history: History):
        """Evaluate global model."""
        num_train_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        train_metrics: Dict[str, Dict[str, float]] = dict()
        num_test_examples: List[int] = []
        test_losses: Dict[str, float] = dict()
        test_metrics: Dict[str, Dict[str, float]] = dict()

        if fl_round == 0:
            #log(INFO, "Evaluating initial global model")
            self.global_model: List[np.ndarray] = self._get_initial_model()

        if self.val_loader:
            random_client = self.client_manager.sample(0.)[0]
            num_instances, loss, eval_metrics = random_client.evaluate(data=self.val_loader, model=self.global_model)
            num_test_examples = [num_instances]
            test_metrics["Server"] = eval_metrics
            test_losses["Server"] = loss
        else:
            for cid, client_proxy in self.client_manager.all().items():
                num_train_instances, train_loss, train_eval_metrics = client_proxy.evaluate(model=self.global_model,
                                                                                            method="train")

                num_train_examples.append(num_train_instances)
                train_losses[cid] = train_loss
                train_metrics[cid] = train_eval_metrics

                num_test_instances, test_loss, test_eval_metrics = client_proxy.evaluate(model=self.global_model,
                                                                                         method="test")
                num_test_examples.append(num_test_instances)
                test_losses[cid] = test_loss
                test_metrics[cid] = test_eval_metrics
        
        history.add_global_train_losses(self.weighted_loss(num_train_examples, list(train_losses.values())))
        history.add_global_train_metrics(self.weighted_metrics(num_train_examples, train_metrics))

        history.add_global_test_losses(self.weighted_loss(num_test_examples, list(test_losses.values())))
        if history.global_test_losses[-1] <= self.best_loss:
            #log(DEBUG, f"Caching best global model, fl_round={fl_round}")
            self.best_loss = history.global_test_losses[-1]
            self.best_epoch = fl_round
            self.best_model = copy.deepcopy(self.global_model)

        history.add_global_test_metrics(self.weighted_metrics(num_test_examples, test_metrics))
        return test_metrics

    def _get_initial_model(self) -> List[np.ndarray]:
        """Get initial parameters from a random client"""
        random_client = self.client_manager.sample(0.)[0]
        client_model = random_client.get_parameters()
        # log(INFO, "Received initial parameters from one random client!")
        return client_model

    def random_evaluate(self, val_loader: Optional[DataLoader] = None,
                 model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None,
                 params: Dict[str, any] = None, method: Optional[str] = "test", 
                 verbose: bool = True, net = None) -> Tuple[
        int, float, Dict[str, float]]:
                     
        if not params or "criterion" not in params:
            params = dict()
            params["criterion"] = torch.nn.MSELoss()

        self.set_parameters(self.model_in, self.global_model)
        test_result = {}

        for key_ in val_loader.keys():
            data = val_loader[key_]
            loss, mse, rmse, mae, r2, nrmse = test(self.model_in, data, params["criterion"], device="cuda")
            metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2, "NRMSE": nrmse, "loss": loss}
            test_result[key_] = metrics 
            if verbose:
                log(INFO, f"[Dataset {key_} Evaluation on {len(data.dataset)} samples] "
                        f"loss: {loss}, mse: {mse}, rmse: {rmse}, mae: {mae}, nrmse: {nrmse}")
        return test_result
    
    def set_parameters(self, net, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
        else:
            net.load_state_dict(parameters.state_dict(), strict=True)
        return net