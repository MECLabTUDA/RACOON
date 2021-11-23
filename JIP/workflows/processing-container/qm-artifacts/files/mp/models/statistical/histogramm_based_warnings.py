import os 
import json 
import math
import numpy as np 
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from mp.utils.feature_extractor import Feature_extractor # pylint: disable=import-error

class histogramm_based_warning():

    def __init__(self) -> None:
        self.path_to_warnings =  os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'hist_based_warnings')

    def load_seg_feature(self,feature):
        '''Loads all feature values for a given feature in order to use them for a histogram
        Args: 
            feature(str): The name of the feature, compare Feature_extractor
            
        Returns(ndarray): An array filled with the values of the single feature '''
        features = []
        feat_extr = Feature_extractor([feature])
        work_path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"],os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"])
        for id in os.listdir(work_path):
            id_path = os.path.join(work_path,id)
            seg_path_short = os.path.join(id_path,'seg')
            seg_features_path = os.path.join(seg_path_short,'features.json')
            feat_vec = feat_extr.read_feature_vector(seg_features_path)
            if not np.isnan(np.sum(np.array(feat_vec))):
                features.append(feat_vec)
        return np.array(features)

class hist_based_warning_slice_dice(histogramm_based_warning):
    # segmentation smoothness 
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(self.path_to_warnings,'slice_dice')
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.path_to_threshholds = os.path.join(self.path,'threshholds.json')
        if os.path.exists(self.path_to_threshholds):
            with open(self.path_to_threshholds,'r') as file:
                self.thresh_dict = json.load(file)

    def compute_threshholds(self,percentiles):
        '''Computes the threshholds for given percentages. Eg The intervalls 
            where 90 percent of the values are contained. Saves the values as a dict
            Eg dict[0.9] = 0.6 can be interpreted that 90 percent of the data lies above 0.6. 
            
        Args: 
            percentiles(list(floats)): The percentiles. For 30 percent use 0.3 and so on. '''
        data = self.load_seg_feature('dice_scores')
        thresh_dict = OrderedDict()
        for percent in percentiles:
            threshhold = self.get_threshhold(data,percent)
            thresh_dict[percent]=threshhold
        self.save_threshholds(thresh_dict)

    def get_threshhold(self,data,percent):
        '''computes a single threshhold, where x percent of the data are contained. 
            Since the slice dice (segmentation smoothness) is the better the higher, 
            only one theshhold needs to be computed
        Args:
            data(ndarray): The array containing the feature values
            percent(float): How much percent should be contained in the intervall
            
        Returns(float): The threshhold'''
        bins = np.arange(np.min(data),1,step=0.001)
        hist, bin_edges = np.histogram(data,bins=bins)
        total_points = np.sum(hist)
        dens = np.array(hist)/total_points
        len_hist = len(hist)
        for i in range(len(hist)):
            weight = np.sum(dens[len_hist-i:len_hist])
            if weight >= percent:
                return bin_edges[len_hist-i]

    def save_threshholds(self,thresh_dict):
        'saves the dictionary of the threshholds at self.path_to_threshholds'
        with open(self.path_to_threshholds) as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        '''Returns in which "percentile" the given segmentation feature value lies.
        A return value of 0.6 means, that the data point would be contained in a range, where
        60 percent of the training data were. So the higher the return value, the less likely the 
        occurence.'''
        num_threshholds = len(self.thresh_dict.items()) + 1 # because we divide by it
        for i,(_,threshhold) in enumerate(self.thresh_dict.items()):
            if seg_feature > threshhold:
                return (num_threshholds-i)/num_threshholds
        return 0

class hist_based_warning_conn_comp(histogramm_based_warning):

    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(self.path_to_warnings,'conn_comp')
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.path_to_threshholds = os.path.join(self.path,'threshholds.json')
        if os.path.exists(self.path_to_threshholds):
            with open(self.path_to_threshholds,'r') as file:
                self.thresh_dict = json.load(file)

    def compute_threshholds(self,percentiles):
        '''Computes the threshholds for given percentages. Eg The intervalls 
            where 90 percent of the values are contained. Saves the values as a dict
            Eg dict[0.9] = 25 can be interpreted that 90 percent of the data lies have less then
            25 connected components. Contrary to slice dice, the threshholds go higher, because most seg
            had a low nuber of conn comp
            
        Args: 
            percentiles(list(floats)): The percentiles. For 30 percent use 0.3 and so on. '''
        data = self.load_seg_feature('conn_comp')
        thresh_dict = OrderedDict()
        for percent in percentiles:
            threshhold = self.get_threshhold(data,percent)
            thresh_dict[percent]=threshhold
        self.save_threshholds(thresh_dict)

    def get_threshhold(self,data,percent):
        '''computes a list of threshholds, where x percent of the data are contained. 
            Since the number of conncted comp cannot be lower then 1, one threshhold suffices
        Args:
            data(ndarray): The array containing the feature values
            percent(float): How much percent should be contained in the intervall
            
        Returns(float): The threshhold'''
        hist, bin_edges = np.histogram(data,bins=np.arange(0,np.max(data),step=1),density=True)
        cum_hist = np.cumsum(hist)
        for i in range(len(cum_hist)):
            if cum_hist[i]>percent:
                return math.ceil(bin_edges[i])

    def save_threshholds(self,thresh_dict):
        with open(self.path_to_threshholds) as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.items()) + 1
        for i,(_,threshhold) in enumerate(self.thresh_dict.items()):
            if seg_feature < threshhold:
                return (num_threshholds-i)/num_threshholds
        return 0

class hist_based_warning_int_mode(histogramm_based_warning):

    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(self.path_to_warnings,'gauss_params')
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        self.path_to_threshholds = os.path.join(self.path,'threshholds.json')
        if os.path.exists(self.path_to_threshholds):
            with open(self.path_to_threshholds,'r') as file:
                self.thresh_dict = json.load(file)

    def compute_threshholds(self,percentiles):
        '''Computes the threshholds for given percentages. Eg The intervalls 
            where 90 percent of the values are contained. Saves the values as a dict
            Eg dict[0.9] = [0.1,0.2,0.5,0.6] can be interpreted that 90 percent of the data lies 
            within [0.1,0.2] and [0.5,0.6]. 
            
        Args: 
            percentiles(list(floats)): The percentiles. For 30 percent use 0.3 and so on. '''
        data = self.load_seg_feature('gauss_params')
        thresh_dict = OrderedDict()
        for percent in percentiles:
            threshholds = self.get_threshhold(data,percent)
            thresh_dict[percent]=threshholds
        self.save_threshholds(thresh_dict)

    def get_threshhold(self,data,percent):
        '''computes a list of threshhold, that correspond to intervalls, where x percent of the data are contained. 
            Threshholds can be a list of 4 or 2 values. 
            This is because the modes of the values are somewhere in the intervall [0,1], so we do that in order to be more precise. 
            When the intervalls overlap we have only 2 values (the absolute upper and lower bound) otherwise we have 4 values per percent
                Args:
                    data(ndarray): The array containing the feature values
                    percent(float): How much percent should be contained in the intervall
                    
                Returns(float): The list of threshholds'''
        #first fit a mixture with 2 components to find 2 modes
        gm = GaussianMixture(n_components=2).fit(data)
        if gm.means_[0][0] < gm.means_[1][0]:
            means = [gm.means_[0][0],gm.means_[1][0]]
            vars = [gm.covariances_[0][0][0],gm.covariances_[1][0][0]]
            weights = [gm.weights_[0],gm.weights_[1]]
            # try to balance the steplengths, according to weights and cov
            step_0 = vars[0]*weights[0]
            step_1 = vars[1]*weights[1]
        else:
            means = [gm.means_[1][0],gm.means_[0][0]]
            vars = [gm.covariances_[1][0][0],gm.covariances_[0][0][0]]
            weights = [gm.weights_[1],gm.weights_[0]]
            # try to balance the steplengths, according to weights and std
            step_0 = (vars[0]**(1/2))*(1/20)*weights[0]
            step_1 = (vars[1]**(1/2))*(1/20)*weights[1]
            
        #find the threshholds
        hist_0, bins_0 = np.histogram(data,np.arange(0,1,step_0))
        hist_1, bins_1 = np.histogram(data,np.arange(0,1,step_1))
        number_points = np.sum(hist_0)
        hist_0 = np.array(hist_0)/number_points
        hist_1 = np.array(hist_1)/number_points
        hist = [hist_0,hist_1]
        bins = [bins_0,bins_1]
        mode_0_bin = np.argmax(bins[0]>means[0])
        mode_1_bin = np.argmax(bins[1]>means[1])
        mode_bins = [mode_0_bin,mode_1_bin]

        # if the intervalls are overlapping, inner intervalls are not increased in this case
        overlapping = False
        complete_0 = False
        complete_1 = False 
        i = 0
        mass = 0
        while mass<percent:
            # check whether intervalls are overlapping 
            if bins[1][mode_bins[1]-i] < bins[0][mode_bins[0]+i+1] and not overlapping:
                #add the bigger bin to the mass
                overlapping = True
                if weights[0]>weights[1]:
                    mass = mass + hist[0][mode_bins[0]+i]
                else:
                    mass = mass + hist[1][mode_bins[1]-i]

            if mode_bins[0]-i < 0 or mode_bins[0]+i > len(hist[0]): 
                complete_0 = True 
            if mode_bins[1]-i < 0 or mode_bins[1]+i > len(hist[1]):
                complete_1 = True
            #if both ditributions have reached their end break the loop
            if complete_1 or complete_0 :
                break
            #add masses
            if i == 0:
                mass = hist[0][mode_bins[0]]+hist[1][mode_bins[1]]
            if overlapping:
                mass = mass + hist[0][mode_bins[0]-i]+hist[1][mode_bins[1]+1]
            else:
                mass0 = hist[0][mode_bins[0]-i]+hist[0][mode_bins[0]+i]
                mass1 = hist[1][mode_bins[1]+i]+hist[1][mode_bins[1]-i]
                mass = mass + mass0 + mass1
            i = i + 1
        if overlapping:
            return [bins[0][mode_bins[0]-i+1],bins[1][mode_bins[1]+i]]
        else:
            return [bins[0][mode_bins[0]-i+1],bins[0][mode_bins[0]+i],bins[1][mode_bins[1]-i+1],bins[1][mode_bins[1]+i]]

    def save_threshholds(self,thresh_dict):
        with open(self.path_to_threshholds) as save_file:
            json.dump(thresh_dict,save_file)

    def label_seg(self,seg_feature):
        num_threshholds = len(self.thresh_dict.items()) +1
        for i,threshholds in enumerate(self.thresh_dict.items()):
            if self.feature_in_threshholds(seg_feature,threshholds):
                return (num_threshholds-i)/num_threshholds
        return 0

    def feature_in_threshholds(self,feature,threshholds):
        if len(threshholds) == 2:
            return ((feature >= threshholds[0]) and (feature <= threshholds[1]))
        if len(threshholds == 4):
            in_first = ((feature >= threshholds[0]) and (feature <= threshholds[1]))
            in_second = ((feature >= threshholds[2]) and (feature <= threshholds[3]))
            return in_first or in_second





    


