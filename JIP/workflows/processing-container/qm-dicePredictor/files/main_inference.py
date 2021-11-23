import os, argparse
from mp.paths import JIP_dir
from mp.quantifiers.IntBasedQuantifier import IntBasedQuantifier
from train_restore_use_models.preprocess_data import preprocess_data
from train_restore_use_models.CNN_train_restore_use import do_inference

# Structure of JIP_dir/data_dirs:
# /
# |---WORKFLOW_DIR --> JIP_dir/data_dirs for inference or JIP_dir/train_dirs for training or JIP_dir/preprocessed_dirs for preprocessed data
#     |---OPERATOR_IN_DIR
#     |   |---0001
#     |   |   |---img
#     |   |   |   |---img.nii
#     |   |   |---seg
#     |   |       |---001.nii
#     |   |       |---002.nii
#     |   |       |---...
#     |   |---0002
#     |   |   |---img
#     |   |   |   |---img.nii
#     |   |   |---seg
#     |   |       |---001.nii
#     |   |       |---002.nii
#     |   |       |---...
#     |   |---...
#     |---OPERATOR_OUT_DIR
#     |---OPERATOR_TEMP_DIR

if __name__ == "__main__": 
    # Build Argumentparser for user to specify the GPU to use 
    parser = argparse.ArgumentParser(description='Train, reterain or use a specified model to predict the quality of CT scans.')
    parser.add_argument('-device', action='store', type=int, nargs=1, default=4,
                        help='Try to train the model on the GPU device with <DEVICE> ID.'+
                            ' Valid IDs: 0, 1, ..., 7'+
                            ' Default: GPU device with ID 4 will be used.')
    parser.add_argument('--preprocess', action='store_const', const=True, default=False,
                        help='Set this flag if preprocessing needs to be done.')
    parser.add_argument('-label', action='store', type=int, nargs=1, default=1,
                        help='Set the label should be relevant for the Dice Predictor.')

    # Extract the GPU
    args = parser.parse_args()
    cuda = args.device
    label = args.label
    preprocess = args.preprocess
    if isinstance(cuda, list):
        cuda = cuda[0]
    if cuda < 0 or cuda > 7:
        assert False, 'GPU device ID out of range (0, ..., 7).'
    cuda = 'cuda:' + str(cuda)
    if isinstance(label, list):
        label = label[0]


    # -------------------------
    # Build environmental vars
    # -------------------------
    print('Building environmental variables..')
    # The environmental vars will later be automatically set by the workflow that triggers the docker container
    # data_dirs (for inference)
    #os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
    #os.environ["OPERATOR_IN_DIR"] = "input"
    #os.environ["OPERATOR_OUT_DIR"] = "output"
    #os.environ["OPERATOR_TEMP_DIR"] = "temp"
    #os.environ["PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent') # pre-trained models

    # preprocessed_dirs (for preprocessed data (output of this workflow = input for main workflow)
    os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
    os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"] = "output_data"
    os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"] = "output_scaled"
    os.environ["PREPROCESSED_OPERATOR_OUT_TRAIN_DIR"] = "output_train"
    os.environ["PREPROCESSED_OPERATOR_OUT_TEST_DIR"] = "output_test"

    # train_dirs (for training data)
    os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')

    # test_dirs (for test data)
    os.environ["TEST_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'test_dirs')

    # Additional arguments for dice predictor
    os.environ["INPUT_FILE_ENDING"] = "nii.gz"
    os.environ["CUDA_FOR_LUNG_SEG"] = cuda
    os.environ["SEED_INTENSITY_SAMPLING"] = '42232323'


    # -- Amins Part -- #
    # Config for artifact classifiers
    config = {'device': cuda, 'input_shape': (1, 60, 299, 299), 'augmentation': True, 'mode': 'inference',
              'data_type': 'inference', 'lr': 1e-3, 'batch_size': 64, 'num_intensities': 5, 'nr_epochs': 250, 'decay_type': 'plat_decay',
              'noise': 'blur', 'weight_decay': 7e-3, 'save_interval': 100, 'msg_bot': False, 'lr_decay': True, 'decay_rate': 0.9,
              'bot_msg_interval': 10, 'nr_images': 25, 'val_ratio': 0.2, 'test_ratio': 0.2, 'augment_strat': 'none',
              'train_on': 'mixed', 'data_augmented': True, 'restore': False, 'store_data': False}
    
    # Preprocess data
    if preprocess:
        preprocessed, error = preprocess_data(config)
        if not preprocessed:
            print('Data could not be preprocessed. The following error occured: {}.'.format(error))

    # Do inference
    inferred, error = do_inference(config)
    if not inferred:
        print('Inference could not be performed. The following error occured: {}.'.format(error))