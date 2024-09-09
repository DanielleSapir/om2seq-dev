from devtools import debug
from dotenv import load_dotenv
load_dotenv('./.env.local')
import sys
import pickle
sys.path.append('/home/danielles/om2seq-dev/src')
from om2seq.data import *
from om2seq.train import *
from utils.wandb_utils import *
from om2seq.benchmarks import *
import pandas as pd
from matplotlib import pyplot as plt
from datasets import Dataset
import cv2
import numpy as np
import glob


def main():
    with open('pickles/S288_train_set.pickle', 'rb') as file:
        train_set = pickle.load(file)

    train_df = pd.DataFrame(train_set)
    new_df = train_df[train_df['query_scale'].isnull() | train_df['reference_id'].isnull() | train_df['orientation'].isnull() | train_df['score'].isnull()]
    reference_id_count = new_df[['reference_id']].isnull().sum()
    orientation_count = new_df[['orientation']].isnull().sum()
    score_count = new_df[['score']].isnull().sum()
    query_scale_count = new_df[['query_scale']].isnull().sum()
    new_df['query_positions_len'] = new_df['BNXLocalizations'].apply(lambda x: len(x) if x is not None else None)
    query_positions_len_1_count = new_df[new_df['query_positions_len'] == 1].shape[0]
    query_positions_len_2_count = new_df[new_df['query_positions_len'] == 2].shape[0]

    new_df_indices = new_df.index
    new_df2 = train_df.loc[new_df_indices]

    pass
    # references = GenomeDataset(limit=self.config.ref_limit).references()

if __name__ == '__main__':
    main()