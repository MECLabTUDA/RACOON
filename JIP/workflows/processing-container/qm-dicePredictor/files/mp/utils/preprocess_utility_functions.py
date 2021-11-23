import shutil
import time
import os 
import SimpleITK as sitk 
import numpy as np 
import torch
import json 
from mp.utils.feature_extractor import Feature_extractor
from mp.utils.Iterators import Dataset_Iterator
from mp.eval.metrics.simple_scores import dice_score
from mp.data.pytorch.transformation import resize_3d
from mp.utils.lung_captured import _extract_lung_segmentation
import multiprocessing as mup

def basic_preprocessing(label=1):
    '''does the basic preprocessing steps of copying the data into the right directory,
    resizing the images, masking the label and computing a segmentation of the lung '''
    copy_data_into_preprocess_dir()
    bring_all_data_into_right_size()
    mask_out_labels_all_seg(label=label)
    compute_lung_segmentations()
    # scale_all_images() # in the paper the images are NOT scaled
    
# COPYING DATA
def copy_data_into_preprocess_dir():
    r"""Copies the data into the preprocess dir in order to iterate over it
    """
    if os.environ["INFERENCE_OR_TRAIN"] == 'inference':
        #get the paths 
        input_path = os.path.join(os.path.sep, os.environ["WORKFLOW_DIR"],os.environ["OPERATOR_IN_DIR"])
        output_path = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        _delete_images_and_labels(output_path)

        #copy the images and segmentations into right format
        ds_iterator = Dataset_Iterator(input_path, mode='JIP')
        ds_iterator.iterate_images(copy_img_seg,preprocess_mode=True)

    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        # get the paths 

        # for NEW DATA FORMAT on JIP platform
        input_path = os.environ["TRAIN_WORKFLOW_DIR"]
        output_path = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        _delete_images_and_labels(output_path)

        #copy the images and segmentations into right format
        for id in os.listdir(input_path):
            if '._' in id or 'DS_Store' in id:
                continue
            start_time = time.time()
            id_path = os.path.join(input_path,id)
            img_path = os.path.join(id_path,'img','img.nii.gz')
            seg_path = os.path.join(id_path,'seg','001.nii.gz')
            copy_img_seg(img_path,seg_path,id)
            for pred in os.listdir(os.path.join(id_path,'pred')):
                if '.DS_Store' in pred:
                    continue
                new_pred_path = os.path.join(output_path,id,'pred',pred)
                if not os.path.isdir(new_pred_path):
                    os.makedirs(new_pred_path)
                shutil.copyfile(os.path.join(id_path,'pred',pred,'pred.nii.gz'),os.path.join(new_pred_path,'pred.nii.gz'))
            end_time = time.time()
            dur = end_time-start_time
            with open('logging_info_private.txt','a') as file: 
                file.write('Copying on {} took {}'.format(id,dur))
                file.write("\r")

        # # For OLD DATA format OLD TRAIN PROCEDURE 
        # output_path = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
        # gt_data = os.path.join(os.path.sep, os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_GT"])
        # _delete_images_and_labels(output_path)

        # #copy the images and segmentations into right format
        # for task in os.listdir(gt_data):
        #     ids = [id.split('_')[0]  for id in os.listdir(os.path.join(gt_data,task,'imagesTr'))]
        #     for id in ids:
        #         start_time = time.time()
        #         img_path = os.path.join(gt_data,task,'imagesTr',id+'_0000.'+os.environ["INPUT_FILE_ENDING"])
        #         seg_path = os.path.join(gt_data,task,'labelsTr',id+'.'+os.environ["INPUT_FILE_ENDING"])
        #         name = task + '_' + id
        #         copy_img_seg(img_path,seg_path,name)
        #         copy_predictions(task,id,name)
        #         end_time = time.time()
        #         dur = end_time-start_time
        #         with open('logging_info_private.txt','a') as file: 
        #             file.write('Copying on {} {} took {}'.format(task,id,dur))
        #             file.write("\r")

def copy_img_seg(img_path,seg_path,name):
    r"""Copies an img and a segmentation into the new folder structure

    Args:
        img_path (str): path to image 
        seg_path (str): path to segmentation
        name (str): string to identify the image-seg pair 
    """
    if os.environ["INFERENCE_OR_TRAIN"] == 'inference':
        save_path_i = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],name,'img')
        save_path_s = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],name,'seg')
    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        save_path_i = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name,'img')
        save_path_s = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name,'seg')
    save_path_img = os.path.join(save_path_i,'img.nii.gz')
    save_path_seg = os.path.join(save_path_s,'001.nii.gz')
    #make directories 
    if not os.path.isdir(save_path_i):
        os.makedirs(save_path_i)
    if not os.path.isdir(save_path_s):
        os.makedirs(save_path_s)
    
    if '.DS_Store' in img_path or '.DS_Store' in seg_path:
        return
    if '._' in img_path or '._' in seg_path:
        return
    
    shutil.copyfile(img_path,save_path_img)
    shutil.copyfile(seg_path,save_path_seg)

def _delete_images_and_labels(path):
    r"""This function deletes every nifti and json (labels) file in the path.

    Args:
        path (str): the path to go through and delete
    """
    # Walk through path and delete all .nii files
    print('Walk trough directory \'{}\' and delete nifti files..'.format(path))
    for dname, dirs, files in os.walk(path):
        for num, fname in enumerate(files):
            msg = str(num + 1) + '_ of ' + str(len(files)) + '_ file(s).'
            print (msg, end = '\r')
            # Check if file is a nifti file and delete it
            if '.nii' in fname or '.json' in fname:
                fpath = os.path.dirname(dname)
                shutil.rmtree(fpath)

def copy_predictions (task,id,name):
    r"""Copies all predictiosn in task, id into the new data format

    Args:
        task (str): The name of the data source
        id (str): the id of the image 
        name (str): a combination of both from above (task_id) which is the destination of the copying process
    """
    pred_data = os.path.join(os.path.sep, os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_PRED"])
    #iterate over all models, that made predictions 
    for model in os.listdir(pred_data):
        
        #look up, if there is a prediction for the img-seg pair
        origin_pred_path = os.path.join(pred_data,model,task,id+'.'+os.environ["INPUT_FILE_ENDING"])
        if os.path.exists(origin_pred_path):
            path_to_id = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name)
            pred_path = os.path.join(path_to_id,'pred',model)

            #copy the prediction
            dst_pred_path = os.path.join(pred_path,'pred.nii.gz')
            if not os.path.isdir(pred_path):
                os.makedirs(pred_path)
            shutil.copyfile(origin_pred_path,dst_pred_path)

## RESIZING 
#now resize images, pred, seg who are too big
def bring_all_data_into_right_size():
    r"""Goes through the working directory and resizes all images necessary """
    work_path = get_workflow_dir()
    for id in os.listdir(work_path):
        if '._' in id or 'DS_Store' in id:
            continue
        start_time=time.time()
        id_path = os.path.join(work_path,id)
        img_path = os.path.join(id_path,'img','img.nii.gz')
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        if check_needs_downsize(img):
           downsize_img_seg_pred(id_path,img) 
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('potential resizing on {} took {}'.format(id,dur))
            file.write("\r")

def check_needs_downsize(img):
    '''checks whether an image is too big and needs resizing 
    Args: 
        img(nd.array): the image
    
    Returns(bool): whether the image is too big
    '''
    shape = np.shape(img)
    if shape[0] > 50 or shape[1] > 512 or shape[2] > 512:
        return True 
    else:
        return False 

def downsize_img_seg_pred(id_path,img):
    '''if the image is too large in dimensions, the image is downsized

    Args: 
        id_path (str): path to the directory where img,seg,pred are
        size (tuple): the size we want to resize to

    Returns 2(nd.array): img,seg resized'''
    #get the right shape 
    shape = np.shape(img)
    new_shape_d = min(50,shape[0])
    new_shape_wh = min(512,shape[1])
    size = (1,new_shape_d,new_shape_wh,new_shape_wh)

    #downsize the image
    img_path = os.path.join(id_path,'img','img.nii.gz')
    img = torch.from_numpy(img)
    img.unsqueeze_(0)
    img = resize_3d(img,size,label=False)
    img = torch.squeeze(img)
    img = img.numpy()
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img,img_path)

    #downsize the segmentation 
    seg_path = os.path.join(id_path,'seg','001.nii.gz')
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    seg = torch.from_numpy(seg)
    seg.unsqueeze_(0)
    seg = seg.type(torch.float64)
    seg = resize_3d(seg,size,label=True)
    seg = torch.squeeze(seg)
    seg = seg.numpy()
    seg = sitk.GetImageFromArray(seg)
    sitk.WriteImage(seg,seg_path)
    
    #downsize possible predictions, its l8, bad code
    all_pred_path = os.path.join(id_path,'pred')
    if os.path.exists(all_pred_path):
        for model in os.listdir(all_pred_path):
            pred_path = os.path.join(id_path,'pred',model,'pred.nii.gz')
            pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            pred = torch.from_numpy(pred)
            pred.unsqueeze_(0)
            pred = pred.type(torch.float64)
            pred = resize_3d(pred,size,label=True)
            pred = torch.squeeze(pred)
            pred = pred.numpy()
            pred = sitk.GetImageFromArray(pred)
            sitk.WriteImage(pred,pred_path) 

## MASKING 
# mask out every label except for 0 (background) and the given label for all segmentations
def mask_out_labels_all_seg(label):
    """Goes through work path and masks out all labels except the one given. 
    All labels are mapped to 0, except label, which is mapped to 1 

    Args:
        arg_1 (type): The label to map to 1 instead of 0
    """
    work_path = get_workflow_dir()
    for id in os.listdir(work_path):
        if '._' in id or 'DS_Store' in id:
            continue
        start_time = time.time()
        seg_path = os.path.join(work_path,id,'seg','001.nii.gz')
        mask_out_label(seg_path,label)
        # now do the same for predictions
        all_pred_path = os.path.join(work_path,id,'pred')
        if os.path.exists(all_pred_path):
            for model in os.listdir(all_pred_path):
                pred_path = os.path.join(all_pred_path,model,'pred.nii.gz')
                mask_out_label(pred_path, label)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('Masking labels on {} took {}'.format(id,dur))
            file.write("\r")

def mask_out_label(seg_path,label):
    """Masks out all labels excpet the one we want to use, which is converted to 1

    Args:
        seg_path (str): path to segmentation
        label (int): label to use
    """
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    seg = np.ma.masked_not_equal(seg,label)
    seg = np.ma.filled(seg,0)
    seg = np.ma.masked_not_equal(seg,0)
    seg = np.ma.filled(seg,1)
    sitk.WriteImage(sitk.GetImageFromArray(seg),seg_path)

##COMPUTING LUNG SEGMENTATIONS
def compute_lung_segmentations():
    r"""Goes through work dir and computes the segmentations for all lung volumes.
        Saves them in the respectie folders
    """
    work_path = get_workflow_dir()
    for id in os.listdir(work_path):
        if '._' in id or 'DS_Store' in id:
            continue
        start_time = time.time()
        compute_lung_segmentation(work_path,id)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('lung segmentation on {} took {}'.format(id,dur))
            file.write("\r")

def compute_lung_segmentation(work_path,id):
    r"""Utility function for compute lung segmentations. It computes the segmentations of the 
        lung volumes given an imahe.

    Args:
        work_path (str): the working dir 
        id (str): the id identifying the image 
    """
    img_path = os.path.join(work_path,id,'img','img.nii.gz')
    lung_seg_path = os.path.join(work_path,id,'lung_seg')
    lung_seg_save_path = os.path.join(lung_seg_path,'lung_seg.nii.gz')
    if not os.path.exists(lung_seg_path):
        os.makedirs(lung_seg_path)
    if torch.cuda.is_available():
        segmentation = _extract_lung_segmentation(img_path, gpu=True, cuda=os.environ["CUDA_FOR_LUNG_SEG"])
    else:
        segmentation = _extract_lung_segmentation(img_path, gpu=False, cuda='cpu')
    segmentation[segmentation==2] = 1
    segmentation = sitk.GetImageFromArray(segmentation)
    sitk.WriteImage(segmentation,lung_seg_save_path)
    
#FEATURE EXTRACTION
def extract_features_all_data():
    r"""Goes through the working dir and computes all features for 
        every image. The features are saved as dictionaries in the respective folders
    """
    work_path = get_workflow_dir()
    feat_extr = Feature_extractor()
    for id in os.listdir(work_path):
        if '._' in id or 'DS_Store' in id:
            continue
        start_time = time.time()
        feat_extr.compute_features_id(id)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('Feat_extr on {} took {}'.format(id,dur))
            file.write("\r")

# TRAINING OF MODELS 
# !!!only for train time!!! compute the dice scores between the predictions and the ground truths
def compute_all_prediction_dice_scores():
    r"""Computes the dice score for all predictions. This is used to train the dice predictor.
    Saves the computed dice scores in the respective folders
    """
    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        work_path = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    else:
        print('this can only be done during train time')
        RuntimeError
    for id in os.listdir(work_path):
        if '._' in id or 'DS_Store' in id:
            continue
        start_time = time.time()
        id_path = os.path.join(work_path,id)
        compute_prediction_dice_scores_for_id(id_path)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('dice scores on predictions on {} took {}'.format(id,dur))
            file.write("\r")

def compute_prediction_dice_scores_for_id(id_path): 
    r"""Recieves the path to an image and computes the dice score for every prediction on it.
    Then saves the computed dice scores 

    Args:
        id_path (str): the path to the folder containing everything related to this id
    """
    seg_path = os.path.join(id_path,'seg','001.nii.gz')
    all_pred_path = os.path.join(id_path,'pred')
    if os.path.exists(all_pred_path):
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        for model in os.listdir(all_pred_path):
            pred_path = os.path.join(id_path,'pred',model,'pred.nii.gz')
            dice_score_save_path = os.path.join(id_path,'pred',model,'dice_score.json')
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            dice = dice_score(seg,prediction)
            with open(dice_score_save_path,'w') as file:
                json.dump(dice,file)

#UTILITY
def get_workflow_dir():
    r"""depending on os.environ["INFERENCE_OR_TRAIN"] return the work dir

    Returns (str): the path to the work dir 
    """
    if os.environ["INFERENCE_OR_TRAIN"] == 'inference':
        work_path = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])
    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        work_path = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    return work_path



#then scale all images
def scale_all_images():
    '''now we can assume, all data is in the same JIP data - format'''
    #set the right directory
    work_path = get_workflow_dir()
    for id in os.listdir(work_path):
        if '._' in id or 'DS_Store' in id:
            continue
        start_time = time.time()
        img_path = os.path.join(work_path,id,'img','img.nii.gz')
        scale_image(img_path)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('Scaling on {} took {}'.format(id,dur))
            file.write("\r")

def scale_image(img_path,d_type=np.float32):
    ''' takes a path to an image, computes the a version with the values scaled to [0,1] 
    and saves it in the same path 
    Args:
        img_path(str): the path to the image
        d_type (np.datatype): a datatype the image shall have as output
    '''
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)

    max_val = np.max(img)
    min_val = np.min(img)
    span = max_val - min_val
    if span == 0:
        print('The image has only one intensity value and thus cannot be rescaled')
        return RuntimeError
        
    shape = np.shape(img)

    add_array = np.ones(shape)*min_val
    img = img - add_array
    img = img * 1/span  
    img = np.around(img,decimals=4)
    if d_type:
        img = np.array(img,dtype=d_type)
    
    img = sitk.GetImageFromArray(img)
    sitk.WriteImage(img,img_path)


# Below unused functions in normal case 

def copy_useable_predictions(task,id,name):
    # DEPRECATED
    global nr_train
    global nr_test
    nr_pred = 0
    pred_data = os.path.join(os.path.sep, os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_PRED"])
    #iterate over all models, that made predictions 
    for model in os.listdir(pred_data):
        
        #look up, if there is a prediction for the img-seg pair
        origin_pred_path = os.path.join(pred_data,model,task,id+'.'+os.environ["INPUT_FILE_ENDING"])
        if os.path.exists(origin_pred_path) and is_useable_prediction(task,model):
            path_to_id = os.path.join(os.path.sep, os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name)
            pred_path = os.path.join(path_to_id,'pred','pred_{}'.format(nr_pred))

            #copy the prediction
            dst_pred_path = os.path.join(pred_path,'pred_'+str(nr_pred)+'.nii.gz')
            if not os.path.isdir(pred_path):
                os.makedirs(pred_path)
            shutil.copyfile(origin_pred_path,dst_pred_path)

            if get_task(task)==7:
                nr_train += 1
            if get_task(task) in [0,1,2,5]:
                nr_test += 1 
def get_task(task):
    #returns the first digit of the task number
    return int(task.split('_')[0][4])

# returns whether there exists a prediction for that specific task
def useable_prediction_exists(task,id):  
    pred_data = os.path.join(os.path.sep, os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_PRED"])
    for model in os.listdir(pred_data):
        #look up, if there is a prediction for the img-seg pair
        origin_pred_path = os.path.join(pred_data,model,task,id+'.'+os.environ["INPUT_FILE_ENDING"])
        if get_task(task) == 7: 
            if get_task(task) == get_task(model) and os.path.exists(origin_pred_path):
                return True
        if get_task(task) in [0,1,2,5] and get_task(model) in [0,1,2,5]:
            if os.path.exists(origin_pred_path):
                return True
    return False
            
def is_useable_prediction(task,model):
    if get_task(task) == 7: 
        return get_task(task) == get_task(model)
    if get_task(task) in [0,1,2,5]:
        return get_task(model) in [0,1,2,5]