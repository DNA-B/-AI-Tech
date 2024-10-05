import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import yaml
import train
import inference
from module.seed_everything import seed_everything

class SeedFold:
    def __init__(self, CFG, seeds: list): 
        self.CFG = CFG
        self.seeds = seeds
        self.n_fold = self.CFG["FOLD_N"]
        self.train_data = pd.DataFrame(pd.read_csv(CFG["TRAIN_PATH"]))
        self.label = self.train_data['label']
        
    def fold(self):
        cnt = 0
        for seed in self.seeds:
            seed_everything(seed)
            kf = StratifiedKFold(n_splits=self.n_fold, random_state=seed, shuffle=True)

            for train_index, dev_index in kf.split(self.train_data, self.label):
                train_df = self.train_data.iloc[train_index]
                dev_df = self.train_data.iloc[dev_index]
                train_df.to_csv(self.CFG["SEED_FOLD_TRAIN_PATH"])
                dev_df.to_csv(self.CFG["SEED_FOLD_DEV_PATH"])
                print("!!!! train, dev split complete !!!!")
                
                train.main()
                inference.main(cnt)
                
                cnt += 1
                print(f'!!! No.{cnt} fold finished !!!')

    def __repr__():
        print("=========== SEED_FOLD FINISHED ===========")

if __name__ == "__main__":     
    with open("/opt/ml/module/config.yaml") as f:
        CFG = yaml.safe_load(f)

    seeds = CFG['SEEDS']
    seed_fold = SeedFold(CFG, seeds)
    seed_fold.fold()
    print(seed_fold)