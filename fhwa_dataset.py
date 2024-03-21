from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, x, y, device='cuda'):
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i): 
        return self.x[i], self.y[i]

def get_shared_dataset(client_id, dataset_name):
    """
    Getting client and shared dataset files by dataset name. 
    Training data is seperated to a small shared group and a large private group.
    """
    dataset_array = {}
    if dataset_name == 'fhwa':
        dataset_path = "FHWA_dataset/torch_dataset/"
    elif dataset_name == 'sumo':
        dataset_path = "/hdd/SUMO_dataset/learn_dataset/"
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    
    public_len = int(0.5*len(dataset_array["train_x"]))
    train_dataset = SequenceDataset(dataset_array["train_x"][public_len:], dataset_array["train_y"][public_len:])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])

    dataset_len = [len(train_dataset), len(val_dataset), len(test_dataset)]
    train_loader_private = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=5, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=5, drop_last=True)

    return train_loader_private, [dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len]], val_loader, test_loader, dataset_len

def get_local_datasets(num_clients, dataset):
    client_dataset = {}
    for c in range(num_clients):
        client_dataset[c] = {}
        train_loader_private, trainset_shared, val_loader, test_loader, dataset_len = get_shared_dataset(c, dataset)
        client_dataset[c]["train_private"] = train_loader_private
        client_dataset[c]["train_shared"] = trainset_shared
        client_dataset[c]["val"] = val_loader
        client_dataset[c]["test"] = test_loader
        client_dataset[c]["len"] = dataset_len
    print("Loaded client dataset.")
    
if __name__ == "__main__":
    get_local_datasets(100, "fhwa")