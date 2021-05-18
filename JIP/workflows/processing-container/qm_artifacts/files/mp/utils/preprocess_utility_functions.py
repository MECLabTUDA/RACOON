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

def basic_preprocessing():
    '''does the 3 basic preprocessing steps of copying the data in the right format 
    scaling the images and 
    resizes the images'''
    copy_data_into_preprocess_dir()
    bring_all_data_into_right_size()
    scale_all_images()
    
#first make functions to copy the data into the right storage format
def copy_data_into_preprocess_dir():
    if os.environ["INFERENCE_OR_TRAIN"] == 'inference':
        #get the paths 
        input_path = os.path.join('/', os.environ["WORKFLOW_DIR"],os.environ["OPERATOR_IN_DIR"])
        output_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])
        _delete_images_and_labels(output_path)

        #copy the images and segmentations into right format
        ds_iterator = Dataset_Iterator(input_path, mode='JIP')
        ds_iterator.iterate_images(copy_img_seg,preprocess_mode=True)

    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        #get the paths 
        output_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
        gt_data = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_GT"])
        _delete_images_and_labels(output_path)

        #copy the images and segmentations into right format
        for task in os.listdir(gt_data):
            ids = [id.split('_')[0]  for id in os.listdir(os.path.join(gt_data,task,'imagesTr'))]
            for id in ids:
                start_time = time.time()
                img_path = os.path.join(gt_data,task,'imagesTr',id+'_0000.'+os.environ["INPUT_FILE_ENDING"])
                seg_path = os.path.join(gt_data,task,'labelsTr',id+'.'+os.environ["INPUT_FILE_ENDING"])
                name = task + '_' + id
                copy_img_seg(img_path,seg_path,name)
                copy_predictions(task,id,name)
                end_time = time.time()
                dur = end_time-start_time
                with open('logging_info_private.txt','a') as file: 
                    file.write('Copying on {} {} took {}'.format(task,id,dur))
                    file.write("\r")

#then scale all images
def scale_all_images():
    '''now we can assume, all data is in the same JIP data - format'''
    #set the right directory
    work_path = get_workflow_dir()
    for id in os.listdir(work_path):
        start_time = time.time()
        img_path = os.path.join(work_path,id,'img','img.nii.gz')
        scale_image(img_path)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('Scaling on {} took {}'.format(id,dur))
            file.write("\r")

#now resize images, pred, seg who are too big
def bring_all_data_into_right_size():
    work_path = get_workflow_dir()
    for id in os.listdir(work_path):
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

#extract the features for the img-seg and img-pred pairs 
def extract_features_all_data():
    work_path = get_workflow_dir()
    feat_extr = Feature_extractor()
    for id in os.listdir(work_path):
        start_time = time.time()
        feat_extr.compute_features_id(id)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('Feat_extr on {} took {}'.format(id,dur))
            file.write("\r")

# !!!only for train time!!! compute the dice scores for the predictions
def compute_all_prediction_dice_scores():
    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    else:
        print('this can only be done during train time')
        RuntimeError
    for id in os.listdir(work_path):
        start_time = time.time()
        id_path = os.path.join(work_path,id)
        compute_prediction_dice_scores_for_id(id_path)
        end_time = time.time()
        dur = end_time-start_time
        with open('logging_info_private.txt','a') as file: 
            file.write('dice scores on predictions on {} took {}'.format(id,dur))
            file.write("\r")

def get_workflow_dir():
    if os.environ["INFERENCE_OR_TRAIN"] == 'inference':
        work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"])
    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
    return work_path

def copy_img_seg(img_path,seg_path,name):
    if os.environ["INFERENCE_OR_TRAIN"] == 'inference':
        save_path_i = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],name,'img')
        save_path_s = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"],name,'seg')
    if os.environ["INFERENCE_OR_TRAIN"] == 'train':
        save_path_i = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name,'img')
        save_path_s = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name,'seg')
    save_path_img = os.path.join(save_path_i,'img.nii.gz')
    save_path_seg = os.path.join(save_path_s,'001.nii.gz')
    #make directories 
    if not os.path.isdir(save_path_i):
        os.makedirs(save_path_i)
    if not os.path.isdir(save_path_s):
        os.makedirs(save_path_s)
    
    shutil.copyfile(img_path,save_path_img)
    shutil.copyfile(seg_path,save_path_seg)

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

def check_needs_downsize(img):
    '''checks whether an image is too big
    Args: 
        img(nd.array): the image
    
    Returns(bool): whether the image is too big
    '''
    shape = np.shape(img)
    if shape[0] > 50 or shape[1] > 512 or shape[2] > 512:
        print(shape)
        return True 
    else:
        return False 

def downsize_img_seg_pred(id_path,img):
    '''if the image was too large in dimensions, the image is downsized
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
        nr_pred = len(os.listdir(all_pred_path))
        for i in range(nr_pred):
            pred_path = os.path.join(id_path,'pred','pred_{}'.format(i),'pred_{}.nii.gz'.format(i))
            pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            pred = torch.from_numpy(pred)
            pred.unsqueeze_(0)
            pred = pred.type(torch.float64)
            pred = resize_3d(pred,size,label=True)
            pred = torch.squeeze(pred)
            pred = pred.numpy()
            pred = sitk.GetImageFromArray(pred)
            sitk.WriteImage(pred,pred_path) 

def _delete_images_and_labels(path):
    r"""This function deletes every nifti and json (labels) file in the path."""
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
    nr_pred = 0
    pred_data = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"],os.environ["TRAIN_WORKFLOW_DIR_PRED"])
    #iterate over all models, that made predictions 
    for model in os.listdir(pred_data):
        
        #look up, if there is a prediction for the img-seg pair
        origin_pred_path = os.path.join(pred_data,model,task,id+'.'+os.environ["INPUT_FILE_ENDING"])
        if os.path.exists(origin_pred_path):
            path_to_id = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"],name)
            pred_path = os.path.join(path_to_id,'pred','pred_{}'.format(nr_pred))

            #copy the prediction

            dst_pred_path = os.path.join(pred_path,'pred_'+str(nr_pred)+'.nii.gz')
            if not os.path.isdir(pred_path):
                os.makedirs(pred_path)
            shutil.copyfile(origin_pred_path,dst_pred_path)
    
def compute_prediction_dice_scores_for_id(id_path): 
    seg_path = os.path.join(id_path,'seg','001.nii.gz')
    all_pred_path = os.path.join(id_path,'pred')
    if os.path.exists(all_pred_path):
        nr_pred = len(os.listdir(all_pred_path))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        for i in range(nr_pred):
            pred_path = os.path.join(id_path,'pred','pred_{}'.format(i),'pred_{}.nii.gz'.format(i))
            dice_score_save_path = os.path.join(id_path,'pred','pred_{}'.format(i),'dice_score.json')
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            dice = dice_score(seg,prediction)
            with open(dice_score_save_path,'w') as file:
                json.dump(dice,file)