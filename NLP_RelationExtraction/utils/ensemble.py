import pandas as pd
import numpy as np
from num_to_label import *
import os

def str_to_lst(probs): 
    result = probs[1:len(probs) - 1].split(',')
    result = list(map(float, result))

    return result

def elementwise_sum(lst1, lst2):
    return list(np.array(lst1) + np.array(lst2))

def elementwise_divide(lst, n):
    return list(np.array(lst) / n)

def elementwise_multiplication(prob, n):
    return list(np.array(prob) * n)

def main(path, weights):
    csvs = os.listdir(path)
    csvs.sort()

    dfs = []

    for csv in csvs:
        dfs.append(pd.read_csv(path+f"/{csv}"))

    probs = [[0.0] * 30 for _ in range(len(dfs[0]))]

    for i in range(len(dfs)):
        for id in range(len(dfs[0])):
            prob = str_to_lst(dfs[i].loc[id,"probs"])
            probs[id] = elementwise_sum(probs[id], elementwise_multiplication(prob, weights[i]))
    
    output_prob = [elementwise_divide(prob, sum(weights)) for prob in probs]

    test_id = dfs[0]["id"]
    pred_answer = num_to_label([np.argmax(np.array(prob)) for prob in output_prob])
    output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob})
    
    output.to_csv(path+"/weighted_ensemble_output.csv", index=False)

if __name__ == "__main__":
    path = "/opt/ml/prediction/weight_ensemble_csv" 
    weights = []
    main(path, weights)