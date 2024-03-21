import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import statsmodels.api as sm
import pandas as pd

from pylab import rcParams
decompositions = []


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
    num_elements = len(predictions[0])
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
    # print(f"Preds shape is: {preds.shape}")
    preds_by_feat = preds[:, :, atk_feature]
    # print(f"Preds shape by feat is: {preds_by_feat.shape}")
    r_a = verify(preds_by_feat)
    # r_a = verify_by_allowable(preds_by_feat, y_true[:, atk_feature])
    r_a = np.asarray(r_a)   
    r_a = np.sum(r_a, axis=1).reshape(-1, 1)
    print(r_a)
    return r_a

# def sat_verifier_all_feats(pred_list, y_true):
#     preds = np.asarray(pred_list)
#     print(f"Preds shape is: {preds.shape}")
#     atk_features = y_true.shape[-1]
#     rets = []
#     for feat_idx in range(atk_features):
#         preds_by_feat = preds[:, :, feat_idx]
#         print(f"Preds shape by feat is: {preds_by_feat.shape}")
#         r_a = verify(preds_by_feat)
#         # r_a = verify_by_allowable(preds_by_feat, y_true[:, atk_feature])
#         r_a = np.asarray(r_a)   
#         r_a = np.sum(r_a, axis=1).reshape(-1, 1)
#         rets.append(r_a)
#         # print(r_a)
#     rets = np.asarray(rets)
#     return rets

def sat_verifier_all_feats(pred_list, y_true):
    preds = np.asarray(pred_list)
    # print(f"Preds shape is: {preds.shape}")
    atk_features = y_true.shape[-1]
    rets = []
    for feat_idx in range(atk_features):
        preds_by_feat = preds[feat_idx, :, :]
        # print(f"Preds shape by feat is: {preds_by_feat.shape}")
        r_a = verify(preds_by_feat)
        # r_a = verify_by_allowable(preds_by_feat, y_true[:, atk_feature])
        r_a = np.asarray(r_a)   
        r_a = np.sum(r_a, axis=1).reshape(-1, 1)
        rets.append(r_a)
        # print(r_a)
    rets = np.asarray(rets)
    return rets
    
def decompose_ts(ts_list, malicious_idxs = [], flr=0, var_idx=0):
    ret_trends = []
    for idx, ts in enumerate(ts_list):
        rcParams['figure.figsize'] = 12, 8
        num_elements = len(ts)
        df = pd.DataFrame(ts)
        start_date = '2024-01-01'
        end_date = pd.to_datetime(start_date) + pd.DateOffset(days=num_elements - 1)
        datetime_index = pd.date_range(start=start_date, end=end_date, freq='D')
        df['Date'] = datetime_index
        df.set_index('Date', inplace=True)
        y=df[0].resample('D').mean()
        decomposition = sm.tsa.seasonal_decompose(y, model='additive')
        fig = decomposition.plot()
        os.makedirs(f"plots/{flr}", exist_ok=True)
        fig.savefig(f"plots/{flr}/{idx}_{'MALICIOUS' if idx in malicious_idxs else ''}_decomposed.png")
        # import IPython
        # IPython.embed()
        ret_trends.append(decomposition.trend.values)
    return ret_trends

def trend_verifier(preds, malicious_idxs = [], flr=0, y_true=None):
    preds = np.asarray(preds)
    total_variations = preds.shape[-1]
    print(preds.shape)
    allfeat_trends = []
    for var_idx in range(total_variations):
        ts_list = preds[:,:, var_idx]
        # print(f"ts_list.shape: {ts_list.shape}")
        trend_list = decompose_ts(ts_list, malicious_idxs, flr, var_idx)
        trend_list = np.asarray(trend_list)
        trend_list = np.nan_to_num(trend_list)
        # print(f"trend_list.shape: {trend_list.shape}")
        allfeat_trends.append(trend_list)
    rets = sat_verifier_all_feats(allfeat_trends, y_true)
    return rets    

if __name__ == "__main__":
    a = np.random.rand(20,10)
    r_a = verify(a)
    r_a = np.asarray(r_a)
    r_a = np.sum(r_a, axis=1)
    # print(f"R_A.shape is: {r_a.shape}")
    print(f"Verification Result is: R_A = {r_a}")
    # print(r_a)