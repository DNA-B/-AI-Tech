import pickle

def num_to_label(label):
    origin_label = []

    with open('/opt/ml/dataset/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    
    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label