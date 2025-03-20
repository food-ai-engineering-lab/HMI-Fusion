import os
import numpy as np
import pandas as pd
import sys
import sys
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from architecture import EfficientNet, SpectralSpatialNet, SpectralPredictor
from datatools import McolonyTestData, SpectralHMITestData
from datatools import McolonyTestData
from sklearn.metrics import precision_score, recall_score, f1_score

def get_label_from_name(name):
    if 'Enter' in name:
        return 'Enter'
    elif 'I4' in name:
        return 'I4'
    elif 'Johan' in name:
        return 'Johan'
    elif 'Kentucky' in name:
        return 'Kentucky'
    elif 'Infant' in name:
        return 'Infant'
    raise ValueError(f'Unknown label for name: {name}')
    
def main(args):    
    # Arguments from command line
    root_train = args.root_train
    root = args.root
    workers = args.workers
    batch = args.batch
    ckpt = args.ckpt
    
    # Set multiprocessing strategy to 'file_system' for pytorch
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Get class names from the training dataset directory
    ds_classes = sorted(os.listdir(root_train))
    
    # Get experimental group name from test dataset directory
    ds_group = root.split('/')[-2] # this reads the first two of each folder's name
    
    # Initialize DataModule   
    # ds = McolonyTestData(root=root)
    ds = SpectralHMITestData(root=root)
    test_dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=workers, pin_memory=False)

    # Load model from checkpoint
    # model = SpectralSpatialNet
    model_args = {'output_len': 1024}
    # model = SpectralSpatialNet.load_from_checkpoint(ckpt, **model_args)
    model = SpectralPredictor.load_from_checkpoint(ckpt, **model_args)
    model.cuda()    # Move model to GPU
    model.eval()    # Set model to evaluation mode
    print('Model loaded')

    itt = iter(test_dl)
    batches = int(np.ceil(len(ds)/8))

    name_list = []
    pred_list = []
    gt_list = []

    for i in tqdm(range(batches)):
        d = next(itt)
        data, fnames = d  # data is (images, masks, spectra)
        
        if isinstance(model, SpectralSpatialNet) or isinstance(model, SpectralPredictor):
            # Move each component to GPU as a batch tensor
            images = data[0].cuda()  # Entire image batch tensor
            masks = data[1].cuda()   # Entire mask batch tensor
            spectra = data[2].cuda() # Entire spectra batch tensor

            input = (images, masks, spectra)
        else:
            input = data['image'].cuda()  # Entire image batch tensor
        # Perform inference
        with torch.no_grad():
            if isinstance(model, SpectralPredictor):
                pred, _ = model(input)
            # get pred for each image
            pred = [torch.argmax(x,dim=0).cpu().numpy() for x in pred]
            pred_list.append(pred) #list, e.g. [[5], [0]]
            name_list.append(fnames)
            print(fnames)

        # Clear GPU memory every 25 batches
        if i % 25 == 0:
            torch.cuda.empty_cache()

    # Post-processing on the prediction and ground truth lists
    pred_list = [float(x) for x in sum(pred_list,[])] #list, e.g. [5.0, 0.0]
    name_list = sum([list(x) for x in name_list],[])
    gt_list = [get_label_from_name(x) for x in name_list]
    gt_list = [get_label_from_name(x) for x in name_list]
    pred_list = [ds_classes[int(x)] for x in pred_list]
    pred_list = [get_label_from_name(x) for x in pred_list]
    # pred_list = [x.split('-')[4][1:] for x in pred_list]
    pred_list = [get_label_from_name(x) for x in pred_list]
    # pred_list = [x.split('-')[4][1:] for x in pred_list]
    
    # Save model inference results as a csv file
    test_df = pd.DataFrame({'filename': name_list, 'gt strain': gt_list, 'pred strain': pred_list})
    test_df.to_csv('microcolony_'+ds_group+'.csv', index=False, header=True)

    # Compute the accuracy
    # accuracy = (test_df['gt strain'] == test_df['pred strain']).mean()
    correct = 0
    for i in range(len(gt_list)):
        print(f"GT: {gt_list[i]}, Pred: {pred_list[i]}")
        if gt_list[i] in pred_list[i]:
            correct += 1
    accuracy = correct / len(gt_list)
    # accuracy = (test_df['gt strain'] == test_df['pred strain']).mean()
    correct = 0
    for i in range(len(gt_list)):
        print(f"GT: {gt_list[i]}, Pred: {pred_list[i]}")
        if gt_list[i] in pred_list[i]:
            correct += 1
    accuracy = correct / len(gt_list)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    
    # Save model inference results as a csv file
    test_df = pd.DataFrame({'filename': name_list, 'gt strain': gt_list, 'pred strain': pred_list})
    test_df.to_csv('microcolony_'+ds_group+'.csv', index=False, header=True)


# mp edits Compute Precision, Recall, and F1 score
    precision = precision_score(test_df['gt strain'], test_df['pred strain'], average='weighted')
    recall = recall_score(test_df['gt strain'], test_df['pred strain'], average='weighted')
    f1 = f1_score(test_df['gt strain'], test_df['pred strain'], average='weighted')

    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1-Score: {f1 * 100:.2f}%')


    # Compute the accuracy
    # accuracy = (test_df['gt strain'] == test_df['pred strain']).mean()
    # print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Generate and save confusion matrix
    # df = test_df
    df = pd.read_csv('microcolony_'+ds_group+'.csv', sep=",")
    confusion_matrix = pd.crosstab(df['gt strain'], df['pred strain'], rownames=['Observed'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix/confusion_matrix.sum(axis=1), annot=True, fmt='.2f', cmap='Blues', annot_kws = {"size": 15})
    plt.savefig('confusion_matrix_0216_epoch33_'+ds_group+'.png', dpi=400)
    plt.show()
if __name__ == '__main__':
    # Define and parse command line arguments
    if len(sys.argv) >1:
        parser = argparse.ArgumentParser()
        parser.add_argument('-rt', '--root_train', type=str, help='Root folder of the training dataset',
                            default='/mnt/projects/bhatta70/VBNC-Detection/data_rgb/train/')
        parser.add_argument('-r', '--root', type=str, help='Root folder of the test dataset', 
                            default='/mnt/projects/bhatta70/VBNC-Detection/data_rgb/test/')
        parser.add_argument('-c', '--ckpt', type=str, help='Path to checkpoint file', required=True)
        parser.add_argument('-w', '--workers', type=int, help='Number of dataloader workers per GPU', default=0)
        parser.add_argument('-b', '--batch', type=int, help='Batch size per GPU', default=1)
        args = parser.parse_args()
    else:
        class Args:
            root_train = '/mnt/projects/bhatta70/HMI-Fusion/data_rgb/train/'
            root = '/mnt/projects/bhatta70/HMI-Fusion/data_rgb/test/'
            ckpt = '/mnt/projects/bhatta70/HMI-Fusion/lightning_logs/my_model/version_41/checkpoints/mcolony-epoch=53-val_loss_epoch=0.47.ckpt'
            workers = 0
            batch = 1
        args = Args()



    main(args)
