## file for functions used in density estimation 
import numpy as np 
import os 
from mp.utils.Iterators import Dataset_Iterator


def get_intensities(list_of_paths, min_size=100, mode='JIP',save = True, save_name='intensities_last_density_train', 
        save_descr='intensities_last_density train', verbose=False):
        '''goes through the given directories and there through every image-segmentation
        pair, in order to sample intensity values from every consolidation bigger 
        then min_size. 
        Assumes, that images have endings as in UK_Frankfurt.

        Args :
                list_of_paths (list(strings)): every string is a path to a directory we want to get intensity values from
                min_size (int): minimal size of component, to get iterated over
                mode (str) :gives the mode, the data is saved, c.f. mp.utils.Iterators.py

        Returns: (ndarray(floats)): a one-dim array of intensity values
        '''

        # Compute number of images to take samples from to adjust sample size. With rising number of images, 
        # small components get more weight, take care, might lead to negative effects
        nr_images = 0
        if list_of_paths:
                for path in list_of_paths:
                        length = len(os.listdir(path))
                        nr_images += length
                if verbose:
                        print('getting intensities for {} images'.format(nr_images))
                number = min(int(150000/length),5000)
        
        list_intesities = []
        for path in list_of_paths:
                ds_iterator = Dataset_Iterator(path,mode=mode)
                samples = ds_iterator.iterate_components(sample_intensities,
                                                threshold=min_size,number=number)
                list_intesities.append(samples)
        arr_intensities = np.array(list_intesities).flatten()

        if save:
                if save_name == None or save_descr == None:
                        print('Not saving due to missing name and or describtion, hod your data clean' )
                else:
                        save_path = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'intensities',save_name+'.npy')
                        if not os.path.exists(os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'intensities')):
                                os.makedirs(os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'intensities'))
                        np.save(save_path,arr_intensities)
                        with open(os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'intensities',save_name+'_descr.txt'),'w') as file:
                                file.write(save_descr)
        
        return arr_intensities

def load_intensities(list_of_names):
        int_path = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'intensities')
        intensities = np.array([])
        for name in list_of_names:
                path = os.path.join(int_path,name+'.npy')
                values = np.load(path)
                intensities = np.append(intensities,values)
        return intensities


def sample_intensities(img,seg,props,number=5000):
        '''samples intesity values from from given component of an img-seg pair
        
        Args:
                img (ndarray): image of intensity values
                seg (ndarray): the respective segmentation mask
                props (list(dict)): the list of the regionprops of the image, 
                        for further documentation see skimage -> regionprops
                number (int): how many samples we want to get
                
        Returns: (list(numbers)): the sampled intensity values'''
                
        coords = props.coords
        rng = np.random.default_rng()
        if len(coords) > number:
                coords = rng.choice(coords,number,replace=False,axis=0)

        intensities = np.array([img[x,y,z] for x,y,z in coords])
        samples = np.random.choice(intensities,number)
        return samples