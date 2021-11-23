import os 
from mp.paths import JIP_dir
import multiprocessing as mup
import torch 

# give random seed for intensity value sampling, to ensure reproducability if somehow
# wanted.
os.environ["SEED_INTENSITY_SAMPLING"] = '42232323'

#set which cuda to use, when lung segmentations are compute
#format is expected to be cuda:<nr_of_cuda> 
os.environ["CUDA_FOR_LUNG_SEG"] = 'cuda:0'

#set environmental variables
#for data_dirs folder, nothing changed compared to Simons version 
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input_small"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

# preprocessing dir and subfolders 
os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"] = "output_scaled"
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = "output_scaled_train"

#dir where train data for intensites is stored (this only needs to be trains_dirs, but since i have more 
# datasets, another subfolder is here)
# os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')
os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs','proper_train_dirs')

#ignore
##below is for christian only, used for older data structures where models are trained on
os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('Covid-RACOON','All images and labels')
os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('Covid-RACOON','All predictions')
# os.environ["TRAIN_WORKFLOW_DIR_GT"] = os.path.join('gt_small')
# os.environ["TRAIN_WORKFLOW_DIR_PRED"] = os.path.join('pred_small')

#which mode is active either 'train' or 'inference' 
os.environ["INFERENCE_OR_TRAIN"] = 'inference'

#ignore
# the ending of the image files in train_dir is only for older datasets
os.environ["INPUT_FILE_ENDING"] = 'nii.gz'

# Whole inference Workflow, metric dict gets output into "output" in "data_dirs"
def inference(label=1):
    os.environ["INFERENCE_OR_TRAIN"] = 'inference'
    from mp.quantifiers.IntBasedQuantifier import IntBasedQuantifier
    quantifier = IntBasedQuantifier(label=label)
    quantifier.get_quality()    

# Train Workflow
def train_workflow(preprocess=True,train_dice_pred=True,verbose=True, label=1):
    os.environ["INFERENCE_OR_TRAIN"] = 'train'
    from train_restore_use_models.train_int_based_quantifier import train_int_based_quantifier
    train_int_based_quantifier(preprocess,train_dice_pred,verbose,label)

def main():
    train_workflow()
    inference()

if __name__ == '__main__' : 
    if not torch.cuda.is_available():
        # On my machine (home Laptop) this command is necessary to avoid an error message
        # while the lung segmentations are computed.
        # Since i expect the server to be able to handle multiprocessing, this will only be 
        # used when there is no server (so no cuda to do multiprocessing).
        mup.freeze_support()
    main()
