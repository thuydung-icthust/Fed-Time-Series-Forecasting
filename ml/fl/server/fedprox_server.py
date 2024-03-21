
from typing import Callable, Dict, List
from torch.utils.data import DataLoader
from ml.fl.server.client_manager import ClientManager
from ml.fl.server.client_proxy import ClientProxy
from ml.fl.server.server import Server


class FedProx_Server(Server):
    def __init__(self, client_proxies: List[ClientProxy], client_manager: ClientManager | None = None, aggregation: str | None = None, aggregation_params: Dict[str, str | int | float | bool] | None = None, weighted_loss_fn: Callable[..., Any] | None = None, weighted_metrics_fn: Callable[..., Any] | None = None, val_loader: DataLoader | None = None, local_params_fn: Callable[..., Any] | None = None, global_val_loaders: DataLoader | None = None, model=None, subval_dataloader=None, defense_on=False, hier_dict=...):
        super().__init__(client_proxies, client_manager, aggregation, 
                         aggregation_params, weighted_loss_fn, 
                         weighted_metrics_fn, val_loader, 
                         local_params_fn, global_val_loaders, 
                         model, subval_dataloader, 
                         defense_on, hier_dict)

        
    