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
    with open('pickles/human_genome.pickle', 'rb') as file:
        human_ref = pickle.load(file)
    with open('pickles/yiest_genome.pickle', 'rb') as file:
        yi_ref = pickle.load(file)
    pass
    # references = GenomeDataset(limit=self.config.ref_limit).references()

if __name__ == '__main__':
    main()