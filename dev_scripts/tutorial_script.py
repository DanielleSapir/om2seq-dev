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

FILE_PATHS = ['aligned_images_dataset.pickle', 'create_dataset_d.pickle', 'dataset_t.pickle']

def get_images():
    directory_path = 'images'
    image_paths = glob.glob(directory_path + '/*')  # Pattern to match all files
    image_dataset = []

    # Loop through all file paths in the directory
    for file_path in image_paths:
        if file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg'):  # Add or remove file types as needed
            # Read the image in grayscale mode
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                image_dataset.append(img[:5, :])
    return image_dataset

def get_tid():
    with open('training_image_ds_first_100.pickle', 'rb') as file:
        tid = pickle.load(file)
    return tid

def get_tds():
    with open('training_dataset.pickle', 'rb') as file:
        tds = pickle.load(file)
    return tds
    
def pickle_dump(a, d, t, file_paths=FILE_PATHS):
    datasets = [a,d,t]
    
    for f, dataset in zip(file_paths, datasets):
        with open(f, 'wb') as file:
            pickle.dump(dataset, file)

def pickle_load(file_paths=FILE_PATHS):
    datasets = []
    for f in file_paths:
        with open(f, 'rb') as file:
        # Deserialize and retrieve the variable from the file
            datasets.append(pickle.load(file))
    return datasets

def create_data():
    a = AlignedImagesDataset(limit=20)
    d = a.create_dataset()
    t = a.dataset()

def history():
    # create_data()
    # pickle_dump(a, d, t)
    # data = pickle_load()
    # a = data[0]
    # d = data[1]
    # t = data[2]
    tds = TrainingDataset(max_workers=0)
    # with open('training_dataset.pickle', 'wb') as file:
    #     pickle.dump(tds, file)
    # with open('training_dataset.pickle', 'rb') as file:
        # tds = pickle.load(file)
    image_ds = tds.training_split['eval'][0:100]['image']
    training_image_ds = tds.training_split['eval'][0:100]
    ref_starts = tds.crops_eval_dataset['ref_start'][0:100]
    ref_stops = tds.crops_eval_dataset['ref_stop'][0:100]
    ref_ids = tds.crops_eval_dataset['reference_id'][0:100]
    # with open('image_ds_first_100.pickle', 'wb') as file:
    #     pickle.dump(image_ds, file)
    #     # tds = pickle.load(file)
    # with open('training_image_ds_first_100.pickle', 'wb') as file:
    #     pickle.dump(training_image_ds, file)
    return tds, image_ds, training_image_ds, ref_starts, ref_stops, ref_ids

def inference(image_dataset):
    inf_model = InferenceModel(model_id_wandb_run_name='89fw8ce7', batch_size=1)
    with torch.inference_mode():
        return [inf_model.inference_y([{'y': image}]) for image in image_dataset]

def get_res(retrievals):
    data = [{'query': ret[0][0],'ref_start': ret[0][1].ref_start, 'ref_stop': ret[0][1].ref_stop, 'reference_id': ret[0][1].reference_id, 'score': ret[0][2].real} for ret in retrievals]
    return pd.DataFrame(data)

def mapping_result(ev, ref: RefEmb, qry: QryEmb, score: float):
        overlap = ev.segment_overlap(ref.ref_start, ref.ref_stop, qry.ref_start, qry.ref_stop)
        return MappingResult(
            qry=qry,
            ref=ref,
            correct=(ref.reference_id == qry.reference_id) and (overlap > 0),
            overlap=overlap,
            score=score,
        )

def add_overlap_column(df, ref_stops, ref_starts):
    # Assuming ref_stops and ref_starts are DataFrames with single columns
    # and their lengths match with df
    stop_min = np.minimum(df['ref_stop'], ref_stops.iloc[:, 0])
    start_max = np.maximum(df['ref_start'], ref_starts.iloc[:, 0])

    df['overlap'] = stop_min - start_max

def show_images_grid(image_list):
    if len(image_list) != 50:
        raise ValueError("The list must contain exactly 50 images.")
    
    rows, cols = 10, 5  # 10 rows, 5 columns
    fig, axes = plt.subplots(rows, cols, figsize=(15, 30))
    axes = axes.flatten()

    for idx, (ax, image) in enumerate(zip(axes, image_list)):
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Index {idx}")
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def main():
    # tds, image_dataset, training_image_ds, ref_starts, ref_stops, ref_ids = history()
    # image_dict = [{'y':[im]}for im in image_dataset]
    # test_set_first_100 = get_tid()
    # tds = get_tds()
    # ref_ids = tds[]
    bmk = Benchmark(enable_om2seq=True, enable_deepom=False, enable_combined=True, model_id_wandb_run_name='89fw8ce7'
                    , num_len=3, ref_limit=None, qry_limit=10, wandb_enabled=False, batch_size=1, num_threads=1, max_workers=0)
    bmk.benchmark_inits()
    test_set = bmk.training_dataset.training_split['train']
    # image_dataset = get_images()
    image_dataset = test_set['image']
    ref_ids = test_set['reference_id']
    
    # # DeepOM evaluation
    # ev_deepom = EvalDeepOM(localizer=bmk.localizer, aligner=bmk.aligner, references=bmk.references, crops=test_set)
    # res = ev_deepom.compute_correctness(test_set)
    # print('DeepOM Evaluation:\nnum correct: ', sum(res),'out of ', len(res), '\naccuracy:', sum(res)/len(res))

    # OM2Seq evaluation
    # image_embs = np.squeeze(inference(image_dataset), axis=1)
    ev = EvalOM2Seq(inference_model=bmk.inference_model,ref_emb_ds=bmk.ref_emb_ds)
    image_embs = ev.inference(test_set)
    metrics = ev.compute_correctness(query_embeddings=image_embs)
    df = pd.DataFrame(metrics)
    df.columns = ['correct']
    mapping_res = ev.mapping_results(query_embeddings=image_embs)
    correct_id = pd.DataFrame([ev.top_result(_).correct_ref_id for _ in mapping_res])
    correct_id.columns = ['correct_id']    
    ds_df = pd.DataFrame(test_set).join(df['correct'])
    ds_df = ds_df.join(correct_id['correct_id'])
    correct_num = ds_df['correct'].value_counts()[True]
    print('OM2Seq Evaluation:\nnum correct:\n', correct_num ,'\nout of ', len(ds_df), '\naccuracy:', correct_num/len(ds_df))
    correct_id_num = ds_df['correct_id'].value_counts()[True]
    print('OM2Seq Evaluation:\nnum correct id:\n', correct_id_num ,'\nout of ', len(ds_df), '\naccuracy:', correct_id_num/len(ds_df))



    # dict(y=None, qry_start=int(qry['QryStartPos']), qry_stop=int(qry['QryEndPos']),\
    #                             crop_image=None, pad_amount=0,\
    #                             crop_orientation=Orientation[XMAPOrientation(str(qry['Orientation'])).name].value, x=None, crop_ref=None, image_scale=ENV.NOMINAL_SCALE, bin_size=int(ENV.NOMINAL_SCALE),\
    #                                   ref_start=int(qry['RefStartPos']), ref_stop=int(qry['RefEndPos']))
    # ret = ev.retrieve(image_embs)
    # # # ref_embs = [ret[i][0][1] for i in range(len(ret))]
    # df = get_res(ret)
    # df['correct_ref_id'] = (ref_ids == df['reference_id'])
    # ref_starts = pd.DataFrame([ref_start for ref_start in test_set['RefStartPos']])
    # ref_stops = pd.DataFrame([ref_stop for ref_stop in test_set['RefEndPos']])
    # add_overlap_column(df, ref_stops, ref_starts)
    # df['correct'] = df['correct_ref_id'] & (df['overlap'] > 0)
    # print(df)
    # print('OM2Seq Evaluation:\nnum correct:\n', df['correct'].value_counts(), '\nout of ', len(df), '\naccuracy:', df['correct'].value_counts()[True]/len(df))

    # Combined evaluation

    ev_combined = EvalCombined(inference_model=bmk.inference_model,
                                ref_emb_ds=bmk.ref_emb_ds,
                                localizer=bmk.localizer,
                                aligner=bmk.aligner,
                                references=bmk.references, crops=test_set)
    ev = EvalOM2Seq(inference_model=bmk.inference_model,ref_emb_ds=bmk.ref_emb_ds, crops=test_set)
    image_embs = ev.inference(test_set)
    # top_result = ev_combined.top_result(mapping_results)
    top_results = [ev_combined.top_result(_).correct for _ in ev.mapping_results(query_embeddings=image_embs)]
    print('accuracy: ', sum(top_results)/len(top_results))
    # save top_results to a file called top_results.pickle
    with open('top_results.pickle', 'wb') as file:
        pickle.dump(top_results, file)
    # mapping_results = [[MappingResult(
    #         qry=ret_ref[0][0],
    #         ref=ret_ref[0][1],
    #         correct=correct,
    #         overlap=overlap,
    #         score=ret_ref[0][2],
    #     )
    #     for ret_ref, correct, overlap  in ref_list] for ref_list in zip(ret,df['correct'],df['overlap'])] #contine
                               
    # bmk_df = bmk.benchmark()
    # bmk.init_om2seq()

    
    print('done')


if __name__ == '__main__':
    main()