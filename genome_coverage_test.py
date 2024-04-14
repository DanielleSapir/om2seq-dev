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

import pandas as pd
import matplotlib.pyplot as plt


def get_coverage_trainins_vs_test(df1, df2):
#   df1 - test set, df2 - train set
#   get a df of the columns "reference_id", 'QryStartPos', 'QryEndPos', 'RefStartPos', 'RefEndPos'
#   for train and test, grouped by reference_id
#   return the coverage of the test set vs the train set
#   df2 = df2.sort_values(by='RefStartPos')
  ref_len = int(df1['RefLen'].values[0])

  test_coverage_start = df1['RefStartPos'].min()
  test_coverage_end = df1['RefEndPos'].max()
  test_coverage_intervals = df1[['RefStartPos', 'RefEndPos']].values.astype(int)
  train_coverage_intervals = df2[['RefStartPos', 'RefEndPos']].values.astype(int)
  
  train_zero_coverage = np.zeros(ref_len)
  test_zero_coverage = np.zeros(ref_len)

  for start,stop in test_coverage_intervals:
    test_zero_coverage[start:stop] = 1
  
  for start,stop in train_coverage_intervals:
    train_zero_coverage[start:stop] = 1
 
  total_overlap = test_zero_coverage*train_zero_coverage
  total_overlap[:int(test_coverage_start)] = 0
  total_overlap[int(test_coverage_end)+1:] = 0
  total_covered_length = total_overlap.sum()
  total_covered_length_test = test_zero_coverage.sum()
  coverage_ratio_from_train = total_covered_length / total_covered_length_test
  detail_dict = {'part of test that is covered in train': coverage_ratio_from_train}
  return detail_dict


def get_coverage_vs_ref(df):
  coverage_intervals = df[['RefStartPos', 'RefEndPos']].values.astype(int)
  ref_len = int(df['RefLen'].values[0])
  coverage_start = df['RefStartPos'].min()
  coverage_end = df['RefEndPos'].max()
  coverage_range = coverage_end - coverage_start
  zero_coverage = np.zeros(ref_len)

  for start,stop in coverage_intervals:
    zero_coverage[start:stop] = 1

  total_covered_length = zero_coverage.sum()
  coverage_ratio_from_reference = total_covered_length / ref_len
  detail_dict = {'coverage_ratio': coverage_ratio_from_reference, 'coverage_range': coverage_range, 'total_covered_length': total_covered_length, 'coverage_start': coverage_start, 'coverage_end': coverage_end}
  return detail_dict

def get_figs(train_set):
    ims = train_set[:10]['image']
    for i, im in enumerate(ims):
        save_im(np.array(im), i, start=100)

def save_im(im, i,start=0):
    im2 = im[:,start:start+250]
    plt.imshow(im2, cmap='gray')
    plt.axis('off')
    plt.savefig(f"Fig1_long_{i}.png", bbox_inches='tight')

def get_xmap_lengths(train_set):
   df = XMAPDataset().dataset().to_pandas()
   df = df[['Confidence', 'RefLen', 'QryLen','HitEnum']]
#    plot the distribution of QryLen vs the confidence score
   filt = (df['QryLen']>=181000) & (df['QryLen']<=768000)
   df_long = df.loc[filt]
   df_long = df_long.sort_values(by='Confidence')
   plt.figure(1, figsize=(10, 6))
   plt.scatter(df_long['Confidence'], df_long['QryLen'], alpha=0.5)
   plt.xlabel('Confidence')
   plt.ylabel('QryLen')
   plt.title('QryLen vs Confidence')
   plt.show()

   df_conf = df.sort_values(by='Confidence', ascending=False)
   df_conf = df_conf[:100000]
   plt.figure(2, figsize=(10, 6))
   plt.scatter(df_conf['Confidence'], df_conf['QryLen'], alpha=0.5)
   plt.xlabel('Top 100k Confidence')
   plt.ylabel('QryLen')
   plt.title('QryLen vs Confidence (top 100k)')
   plt.show()

   filt = (df_conf['QryLen']>=181000) & (df_conf['QryLen']<=768000)
   df_conf_long = df_conf.loc[filt]
   df_conf_long = df_conf_long.sort_values(by='Confidence')
   plt.figure(3, figsize=(10, 6))
   plt.scatter(df_conf_long['Confidence'], df_conf_long['QryLen'], alpha=0.5)
   plt.xlabel('Confidence (top 100k)')
   plt.ylabel('QryLen')
   plt.title('QryLen (in training range) vs Confidence (top 100k)')
   plt.show()

#    df_conf_long['M_Count'] = df_conf_long['HitEnum'].apply(lambda x: x.count('M'))
#    df_conf_long['label_num/length'] = df_conf_long['M_Count']/df_conf_long['QryLen']
#    print('the average number of M labels per length in the training range: ', df_conf_long['label_num/length'].mean())
#    df_conf_long['bp pair per label'] = df_conf_long['QryLen']/df_conf_long['M_Count']
#    print('the average number of bp pairs per label in the training range: ', df_conf_long['bp pair per label'].mean())

   print('number of top 100k molecules in the training range: ', len(df_conf_long))
   print('max confidence from top 100k: ', df_conf['Confidence'].max())
   print('min confidence from top 100k: ', df_conf['Confidence'].min())

   train_set_cp = train_set.to_pandas()
#    train_set_cp['M_Count'] = train_set_cp['HitEnum'].apply(lambda x: x.count('M'))
   train_set_cp['label_num/length'] = train_set_cp['NumberofLabels']/train_set_cp['QryLen']
   print('the average number of labels per length in the training set: ', train_set_cp['label_num/length'].mean())
   train_set_cp['bp pair per label'] = train_set_cp['QryLen']/train_set_cp['NumberofLabels']
   print('the average number of bp pairs per label in the training set: ', train_set_cp['bp pair per label'].mean())

   print('done')


def main():
    bmk = Benchmark(enable_om2seq=True, enable_deepom=True, enable_combined=True, model_id_wandb_run_name='89fw8ce7'
                    , ref_limit=None, qry_limit=None, wandb_enabled=False)
    bmk.benchmark_inits()
    df = bmk.compute_metrics()
    train_set = bmk.training_dataset.training_split['train']
    get_figs(train_set)
    #  get a df of the columns "reference_id", 'QryStartPos', 'QryEndPos', 'RefStartPos', 'RefEndPos'
    train_set_relevant = train_set.select_columns(['reference_id', 'QryStartPos', 'QryEndPos', 'RefStartPos', 'RefEndPos', 'QryLen', 'RefLen', 'HitEnum', 'query_indices', 'NumberofLabels'])
    get_xmap_lengths(train_set_relevant)
    df = pd.DataFrame(train_set_relevant)
    df = df.sort_values(by='reference_id')
    ref_ids = df['reference_id'].unique()
    print('The unique reference ids in the train set are: ', ref_ids)

    test_set = bmk.training_dataset.training_split['test']
    #  get a df of the columns "reference_id", 'QryStartPos', 'QryEndPos', 'RefStartPos', 'RefEndPos'
    test_set_relevant = test_set.select_columns(['reference_id', 'QryStartPos', 'QryEndPos', 'RefStartPos', 'RefEndPos', 'QryLen', 'RefLen'])
    test_df = pd.DataFrame(test_set_relevant)
    
    test_df = test_df.sort_values(by='reference_id')
    test_ref_ids = test_df['reference_id'].unique()
    print('The unique reference ids in the test set are: ', test_ref_ids)
    
    chromosome_coverage_details = {}
    test_coverage_details = {}
    # Iterate through each group in df by 'reference_id'
    for (name_test, group_test), (name_train, group_train) in zip(test_df.groupby('reference_id'), df.groupby('reference_id')):
        print('name_train: ', name_train, 'name_test: ', name_test)
        curr_coverage_test = get_coverage_trainins_vs_test(group_test.copy(), group_train.copy())
        test_coverage_details[name_train] = curr_coverage_test
        print(f'Coverage details for {name_train}: {curr_coverage_test}')

        curr_coverage_train = get_coverage_vs_ref(group_train.copy())
        chromosome_coverage_details[name_train] = curr_coverage_train   
        print(f'Coverage details for {name_train}: {curr_coverage_train}')

    test_train_df = pd.DataFrame(test_coverage_details)
    test_train_df_T = test_train_df.T
    test_train_df_T.to_csv('test_train_coverage.csv')
    chromosome_coverage_df = pd.DataFrame(chromosome_coverage_details)
    chromosome_coverage_df_T = chromosome_coverage_df.T
    chromosome_coverage_df_T.to_csv('chromosome_coverage.csv')
    # for name, group in test_df.groupby('reference_id'):
        # curr_coverage = get_coverage(group.copy())
        # chromosome_coverage_details[name] = curr_coverage   
        # print(f'Coverage details for {name}: {curr_coverage}')
    

    print('done')




if __name__ == '__main__':
    main()