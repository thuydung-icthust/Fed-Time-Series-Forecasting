import numpy as np
import torch
from tqdm import tqdm

def get_local_range(preds):
    np_preds = np.asarray(preds)
    return np.max(preds), np.min(preds)

def get_range_by_interval(preds, interval_size=5):
    total_timesteps = len(preds)
    total_intervals = int(total_timesteps // interval_size)
    if total_timesteps % interval_size != 0:
        total_intervals += 1
    LBs = np.zeros(total_timesteps)
    UBs = np.zeros(total_timesteps)

    for i in range(total_intervals):
        start_idx = i * interval_size
        end_idx = min((i + 1) * interval_size, total_timesteps)
        print(f"end_idx: {end_idx}")
        # Ensure the end index is within the total timesteps
        # end_idx = min(end_idx, total_timesteps)

        interval_preds = preds[start_idx:end_idx]

        # Calculate local range for the interval
        local_max, local_min = get_local_range(interval_preds)
        UBs[start_idx:end_idx] = local_max
        LBs[start_idx:end_idx] = local_min
    return LBs, UBs

def rule_aggregation(LBs, UBs):
    # Aggregate LBs and UBs separately
    # import IPython
    # IPython.embed()
    aggregated_lb = np.median(LBs, axis=0)
    aggregated_ub = np.median(UBs, axis=0)
    # print(f"UBs: {UBs}")
    # print(f"Aggregated_lb: {aggregated_lb.shape}, \nAggregated_ub: {aggregated_ub.shape}")
    
    return aggregated_lb, aggregated_ub

def verify(preds):
    total_clients = len(preds)
    LBs, UBs = [], []
    for i in range(total_clients):
        LB, UB = get_range_by_interval(preds[i])
        LBs.append(LB)
        UBs.append(UB)
    aggregated_lb, aggregated_ub = rule_aggregation(LBs, UBs)
    # import IPython
    # IPython.embed()
    verifier_results = get_satisfied_elements(preds, np.asarray(aggregated_lb), np.asarray(aggregated_ub))
    return verifier_results

def get_allowable_by_interval(y_true, alpha, interval_size=20):
    total_timesteps = len(y_true)
    total_intervals = int(total_timesteps // interval_size)
    if total_timesteps % interval_size != 0:
        total_intervals += 1
    LBs = np.zeros(total_timesteps)
    UBs = np.zeros(total_timesteps)
    MAX_VAL = max(y_true)
    MIN_VAL = min(y_true)
    for i in range(total_intervals):
        start_idx = i * interval_size
        end_idx = min((i + 1) * interval_size, total_timesteps)
        print(f"end_idx: {end_idx}")
        # Ensure the end index is within the total timesteps
        # end_idx = min(end_idx, total_timesteps)

        interval_preds = y_true[start_idx:end_idx]

        # Calculate local range for the interval
        local_max, local_min = get_local_range(interval_preds)
        UBs[start_idx:end_idx] = min(MAX_VAL, local_max + local_max*alpha)
        LBs[start_idx:end_idx] = max(MIN_VAL, local_min - local_min*alpha)
    return LBs, UBs

def verify_by_allowable(preds, y_true):
    total_clients = len(preds)
    # LBs, UBs = [], []
    # for i in range(total_clients):
    #     LB, UB = get_range_by_interval(preds[i])
    #     LBs.append(LB)
    #     UBs.append(UB)
    aggregated_lb, aggregated_ub = get_allowable_by_interval(y_true, alpha=0.05)
    # import IPython
    # IPython.embed()
    verifier_results = get_satisfied_elements(preds, np.asarray(aggregated_lb), np.asarray(aggregated_ub))
    return verifier_results

def pseudo_verify(preds, glob_pred):
    aggregated_lb, aggregated_ub = get_range_by_interval(glob_pred)
    verifier_results = get_satisfied_elements(preds, aggregated_lb, aggregated_ub)
    return verifier_results

def get_satisfied_elements(predictions, lb, ub):
    verifier_results = []
    for j in range(predictions.shape[0]):
        pred = predictions[j]
        # import IPython
        # IPython.embed()
        satisfied_elements = [1.0 if lb[i] <= pred[i] <= ub[i] else 0.0 for i in range(len(pred))]
        verifier_results.append(satisfied_elements)
    return verifier_results

def sat_defense(net_list, val_loader, device):
    preds = []
    for net in net_list:
        pred = []
        net.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Testing", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                pred.extend(outputs.cpu().detach().numpy())
        # print(f"len(pred): {len(pred)}")
        preds.append(pred)
    # global preds:
    preds = np.asarray(preds)
    glob_pred = []
    net.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            glob_pred.extend(outputs.cpu().detach().numpy())
    glob_pred = np.asarray(glob_pred)
    # print(f"preds: {preds.shape}")
    # print(f"preds: {preds}")
    # r_a = verify(preds)
    r_a = pseudo_verify(preds, glob_pred)
    # print(f"preds: {preds.shape}")
    # print(f"preds: {preds}")
    # r_a = verify(preds)
    print(f"r_a: {r_a}")
    r_a = np.asarray(r_a)
    r_a = np.sum(r_a, axis=1).reshape(-1, 1)
    # scaler = MinMaxScaler()
    # Fit and transform the data using the scaler
    # contribution_score = scaler.fit_transform(r_a)

    # If you want the result as a list
    contribution_score_list = r_a.flatten().tolist()
    contribution_score_list = [score/sum(contribution_score_list) for score in contribution_score_list]
    print(f"contribution_score_list: {contribution_score_list}")
    return contribution_score_list

def sat_verifier(pred_list, y_true, atk_feature=0):
    preds = np.asarray(pred_list)
    print(f"Preds shape is: {preds.shape}")
    preds_by_feat = preds[:, :, atk_feature]
    print(f"Preds shape by feat is: {preds_by_feat.shape}")
    # r_a = verify(preds_by_feat)
    r_a = verify_by_allowable(preds_by_feat, y_true[:, atk_feature])
    r_a = np.asarray(r_a)   
    r_a = np.sum(r_a, axis=1).reshape(-1, 1)
    print(r_a)
    return r_a
    
if __name__ == "__main__":
    a = np.random.rand(20,10)
    r_a = verify(a)
    r_a = np.asarray(r_a)
    r_a = np.sum(r_a, axis=1)
    # print(f"R_A.shape is: {r_a.shape}")
    print(f"Verification Result is: R_A = {r_a}")
    # print(r_a)