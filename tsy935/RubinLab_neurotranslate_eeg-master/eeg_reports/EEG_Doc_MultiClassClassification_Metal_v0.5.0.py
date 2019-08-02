import numpy as np
import sys
import os
import re
import pandas as pd
import metal

from eeg_utils import parse_eeg_docs_multiclass
from eeg_utils import EEGNote
from eeg_utils import get_empty_docs
from eeg_utils import create_data_split_multiclass
from eeg_lfs_multiclass import *
from eeg_utils import get_section_with_name
from metal.analysis import lf_summary
from metal.analysis import single_lf_summary, confusion_matrix
from metal.multitask.task_graph import TaskHierarchy
from metal.tuners import RandomSearchTuner
from metal.multitask import MTLabelModel
from scipy.sparse import csr_matrix
import dask
from dask.diagnostics import ProgressBar
from eeg_utils import evaluate_lf_on_docs, create_label_matrix
import pickle
import dill

os.environ['CUDA_VISIBLE_DEVICES']='0'

print('metal version:{}'.format(metal.__version__))

# Setting data location
#eeg_data_path = '/Users/siyitang/Documents/RubinLab/Project/EEG/Data/Reports'
eeg_data_path = '/home/tsy935/Docs/RubinLab/Data/EEG/Reports'
eeg_data_file = 'reports_unique_for_hl_mm_multiclassLabels.csv'
data_path = os.path.join(eeg_data_path, eeg_data_file)

# Set random seed
np.random.seed(123)

# Save dir
SAVE_DIR = '/home/tsy935/output/eeg_reports'

# Parsing documents -- note that 1 = abnormal, 2 = other abnormalities, 3 = not applicable
column_names = ['multiclass_label_abnormal','multiclass_label_seizure', 'multiclass_label_slowing', 
                'multiclass_label_spikes', 'multiclass_label_sharps', 'multiclass_label_suppression', 
                'multiclass_label_discont','multiclass_label_hypsar', 'multiclass_label_seizure_motor', 
                'multiclass_label_seizure_hyperkinetic',
               'multiclass_label_seizure_clonic', 'multiclass_label_seizure_tonic']

# LFs
lfs = [
    [lf_abnormal_interp,
    lf_findall_abnormal_interp,
    lf_normal_interp],   
    
    
    [lf_abnormal_interp_with_seizure,
    lf_findall_interp_with_seizure,
    lf_seizure_section],

    
    [lf_abnormal_interp_with_slowing,
    lf_findall_interp_with_slowing,
    lf_slowing_in_impression],
    
    [lf_abnormal_interp_with_spikes,
    lf_findall_interp_with_spikes,
    lf_spikes_in_impression],
    
    [lf_abnormal_interp_with_sharps,
    lf_findall_interp_with_sharps,
    lf_sharps_in_impression],
    
    [lf_abnormal_interp_with_suppression,
    lf_findall_interp_with_suppression,
    lf_suppression_in_impression],
    
    [lf_abnormal_interp_with_discont,
    lf_findall_interp_with_discont,
    lf_discont_in_impression],
    
    [lf_abnormal_interp_with_hypsarrhythmia,
    lf_findall_interp_with_hypsarrhythmia,
    lf_hypsarrhythmia_in_impression],
    
    [lf_seizure_motor,
    lf_findall_seizure_motor,
    lf_seizure_motor_in_sections],
    
    [lf_seizure_hyperkinetic,
    lf_findall_seizure_hyperkinetic,
    lf_seizure_hyperkinetic_in_sections],
    
    [lf_seizure_clonic,
    lf_findall_seizure_clonic,
    lf_seizure_clonic_in_sections],
    
    [lf_seizure_tonic,
    lf_findall_seizure_tonic,
    lf_seizure_tonic_in_sections]]


def main(column_names, lfs, save_dir):
    # Loading data
    df_eeg = pd.read_csv(data_path, index_col=0).dropna(how='all')

    # Parsing EEG notes
    eeg_note_dill = 'parsed_eeg_notes.dill'
    eeg_note_dill_path = os.path.join(eeg_data_path, eeg_note_dill)

    if os.path.exists(eeg_note_dill_path):
        print('Loading pre-parsed EEG notes...')
        with open(eeg_note_dill_path, 'rb') as af:
            docs = dill.load(af)
    else:
        print('Parsing EEG notes...')
        docs = parse_eeg_docs_multiclass(df_eeg, column_names, use_dask=False)
        with open(eeg_note_dill_path,'wb') as af:
            dill.dump(docs, af)

    # These are docs with empty sections -- most look like they're not EEG reports!   
    empty_docs = get_empty_docs(docs)

    # Removing empty EEG docs
    eeg_docs = list(set(docs)-set(empty_docs))
    print('Number of EEG Reports with Sections: {}'.format(len(eeg_docs)))


    # Shuffling and setting seed
    np.random.shuffle(eeg_docs)

    # Creating data split
    train_docs, dev_docs, test_docs = create_data_split_multiclass(eeg_docs, column_names)
    docs_list = [train_docs, dev_docs, test_docs]

    Y_dev = []
    Y_test = []

    # Computing dev/test label balance for each class
    for i, col_name in enumerate(column_names):
        curr_Y_dev = np.array([doc.gold_label[col_name] for doc in dev_docs])
        curr_Y_test = np.array([doc.gold_label[col_name] for doc in test_docs])
        Y_dev.append(curr_Y_dev)
        Y_test.append(curr_Y_test)

    dev_balance = np.zeros((len(column_names)))
    test_balance = np.zeros((len(column_names)))

    for i in range(len(column_names)):
        dev_balance[i] = np.sum(Y_dev[i] == 1) /len(Y_dev[i])
        test_balance[i] = np.sum(Y_test[i] == 1)/len(Y_test[i])

    NUM_TASKS = len(column_names)
    print('Number of tasks:{}'.format(NUM_TASKS))

    # Resetting LFs
    clobber_lfs = True
    Ls_file = os.path.join(save_dir,'Ls_0p3.pkl')
    Ys_file = os.path.join(save_dir,'Ys_0p3.pkl')

    # Get lf names
    #lf_names = [lf.__name__ for lf in lfs]
    lf_names = [[lf.__name__ for lf in curr_lfs] for curr_lfs in lfs]

    # Loading Ls if they exist

    Ls = [] # Ls is a t-dim list of (n,m) sparse matrices
    Ys = [] # Ys is a t-dim list of (n,) ground truth labels
    if clobber_lfs or (not os.path.exists(Ls_file)):
        print('Computing label matrices...')
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            split_Ls = []
            for j in range(NUM_TASKS):
                curr_Ls = create_label_matrix(lfs[j],docs)
                split_Ls.append(curr_Ls)
            #for j in range(NUM_TASKS):           
            #    split_Ls.append(curr_Ls)
            Ls.append(split_Ls)
            
        with open(Ls_file,'wb') as af:
            pickle.dump(Ls, af)
    
    else:
        print('Loading pre-computed label matrices...')
        with open(Ls_file,'rb') as af:
            Ls=pickle.load(af) 
    
    # Create Ys
    print('Creating label vectors...')
    Ys = [[],Y_dev, Y_test]
    with open(Ys_file,'wb') as af:
        pickle.dump(Ys, af)


    # edges: which task is which task's parent?
    cardinalities = [2] + [2] * (NUM_TASKS-1)
    print('Cardinalities: {}'.format(cardinalities))
    #edges = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
    #         (0,7),(0,8),(0,9),(0,10),(0,11)]
    
    #task_graph = TaskHierarchy(cardinalities=cardinalities, edges=edges)

    # Creating metal label model
    #label_model = MTLabelModel(task_graph=task_graph)
    label_model = MTLabelModel(K=cardinalities)

    # Creating search space
    #search_space = {
    #        'l2': {'range': [0.0001, 0.1], 'scale':'log'},           # linear range
    #        'lr': {'range': [0.0001, 0.01], 'scale': 'log'},  # log range
    #       }

    #searcher = RandomSearchTuner(MTLabelModel(task_graph=task_graph), log_dir='./run_logs',
    #                            log_writer_class=None)
    
    # Train label model
    label_model.train_model(Ls[0], n_epochs=200, log_train_every=20, seed=123)
    
    Y_train_ps = label_model.predict_proba(Ls[0])
    
    # Save trained label model
    Y_train_ps_file = os.path.join(save_dir, 'Y_train_ps')
    with open(Y_train_ps_file, 'wb') as f:
        pickle.dump(Y_train_ps, f)
    label_model_file = os.path.join(save_dir, 'label_model')
    with open(label_model_file, 'wb') as f2:
        pickle.dump(label_model, f2)
    

    print('Score on dev set: {}'.format(label_model.score((Ls[1], Ys[1]))))
    
    

        

if __name__ == '__main__':
    main(column_names, lfs, SAVE_DIR)
