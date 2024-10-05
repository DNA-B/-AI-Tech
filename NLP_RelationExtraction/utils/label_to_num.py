import pickle

def label_to_num(label):
    num_label = []
    with open('/opt/ml/dataset/dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])
  
    return num_label