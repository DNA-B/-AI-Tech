import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

def train_val_split(dataset, ratio: float):
    if ratio == 0.0:
        ratio = 0.01

    lbl_dict = defaultdict(int)

    for lbl in dataset['label']:
        lbl_dict[lbl] += 1

    train_dataset = []
    val_dataset = []

    for item in tqdm(lbl_dict.items(), desc='train_validation_split', total=len(lbl_dict)):
        sub_dataset = dataset[dataset['label'] == item[0]]

        train_data, validation_data = train_test_split(sub_dataset, test_size=ratio, random_state=14)

        train_dataset.append(train_data)
        val_dataset.append(validation_data)

    train_dataset = pd.concat(train_dataset, ignore_index=True)
    val_dataset = pd.concat(val_dataset, ignore_index=True)

    train_dataset = train_dataset.sample(frac=1, random_state=14).reset_index(drop=True)
    val_dataset = val_dataset.sample(frac=1, random_state=14).reset_index(drop=True)

    return train_dataset, val_dataset