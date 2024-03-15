# import os
# import json
# import yaml
# import torch
# import pickle
import logging
import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import multiprocessing
from functools import partial
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
# from eni.ml.magnetic_fusion.mf_disruption.metrics import tpr_score
# from eni.ml.magnetic_fusion.mf_disruption.metrics import fpr_score
# from eni.ml.magnetic_fusion.mf_disruption.metrics import compute_AUC_Zhu

try:
    logger = logging.getLogger('MagneticFusionTrain')
except:
    logger = None



class model_performance():
    def __init__(
            self,
            config=None):
        self.config = config

        
        self.metrics_dict = {
            'f1_score': f1_score,
            'recall_score': recall_score,
            'precision_score': precision_score,
            'balanced_accuracy_score': balanced_accuracy_score,
            'accuracy_score': accuracy_score,
            'roc_auc_score': roc_auc_score,
            'f2_score': partial(fbeta_score, beta=2),
            'f0.5_score': partial(fbeta_score, beta=0.5),
            'confusion_matrix': confusion_matrix,
            'tpr': tpr_score,
            'fpr': fpr_score,
            'AUC_zhu':compute_AUC_Zhu
        }



    # -----------------------------------------------
    def eval(
            self,
            metrics:list,
            unrolled_proba:dict,
            params_dict:dict,
            verbose:bool=True
            ):
        """
        Compute metrics on the shots given the thresholded statistics.
        """
        
        # from unrolled probabilities to [0,1] sequences
        self.unrolled_proba_to_shot_classif(unrolled_proba,params_dict)

        # calc metrics
        self.calc_metrics_report(
            metrics=metrics,
            unrolled_proba=unrolled_proba)
        
        # Print metrics
        if verbose:
            for k,v in self.metrics_report.items():
                logger.info(f'{k} - {np.round(v, 5)}')

        return self.metrics_report



    # -----------------------------------------------
    def calc_metrics_report(
            self,
            metrics:list,
            unrolled_proba:dict=None,
            ):
        '''
        Loops over metrics dictuinary items and computes a metrics_report.
        '''

        self.metrics_report = {}
        for metric_name in metrics:
            if metric_name != 'AUC_zhu':
                self.metrics_report.update(
                    {metric_name:self.metrics_dict[metric_name](self.class_shots_true,self.class_shots_pred)}
                )
            else:
                self.metrics_report.update(
                    {metric_name:self.metrics_dict[metric_name](unrolled_proba,logger)}
                )




    # -----------------------------------------------
    def unrolled_proba_to_shot_classif(
            self,
            unrolled_proba:dict,
            params_dict:dict,
            ):
        '''
        From unrolled probabilities to shot classification using two-threshold+hysteresis time rule.
        '''
        
        self.class_shots_pred = []
        self.class_shots_true = []

        # with multiprocessing.Pool() as pool:
        #     pool.map(task, unrolled_proba.items())

        for ID,shot in unrolled_proba.items():
            class_shots_pred, class_shots_true = self.classify_shot(**shot, **params_dict)
            self.class_shots_pred.append(class_shots_pred)
            self.class_shots_true.append(class_shots_true)

        self.class_shots_pred = np.array(self.class_shots_pred)
        self.class_shots_true = np.array(self.class_shots_true)



    # -----------------------------------------------
    def classify_shot(
            self,
            proba_shot, 
            # y, 
            time_untill_disrupt, 
            time_shot, 
            label_shot, 
            high_thr, 
            low_thr, 
            t_hysteresis,
            t_useful):
        '''
        single shot version of unrolled_proba_to_shot_classif. From unrolled probabilities 
        to shot classification using two-threshold+hysteresis time rule.
        '''

        class_shots_true = label_shot
        if class_shots_true == 1: #cut last t_useful part of the shot, otherwise late a alarm would be a true positive
            proba_shot = proba_shot[time_untill_disrupt>=t_useful]
            time_shot = time_shot[time_untill_disrupt>=t_useful]
        
        # convert t_hysteresis [s] into hysteresis [number of points] by dividing time_step/t_hysteresis
        if t_hysteresis == 0:
            hysteresis = 0
        else:
            hysteresis = (time_shot.ravel()[1]-time_shot.ravel()[0]).round(4)//t_hysteresis
            
        if high_thr == low_thr: 
            # disable hysteresis and look at the entire timeseries at once
            # logger.info('Found lower_thresh = high_tresh -> disabling hysteresis policy')
            class_shots_pred = 1 if (proba_shot >= high_thr).any() else 0

        else:

            # for speeding-up the computation: if none of proba_shot is above high_thr, 
            # label the shot as non-disruptive, else loop over all the timeseries points
            if (proba_shot < high_thr).all():
                class_shots_pred = 0
            else:
                hysteresis_counter = 0 # initialize hysteresis counter
                class_shots_pred = 0
                for i in range(proba_shot.shape[0]):
                    if proba_shot[i] >= high_thr:
                        hysteresis_counter += 1
                    elif proba_shot[i] < low_thr:
                        hysteresis_counter = 0

                    if hysteresis_counter > hysteresis:
                        class_shots_pred = 1
                        break

        return class_shots_pred, class_shots_true


'''
    # -----------------------------------------------
    def compute_thresholded_statistics(
        test_unrolled_predictions, 
        high_threshold, 
        low_threshold,
        hysteresis
    ):

        """
        Compute statistics for thresholded disruptivity warnings.
        In this case, if the model predicts a disruption above the high threshold,
        for a certain number of steps, it is considered a disruption. However,
        if the prediction falls below the low threshold, the hysteresis counter is
        reset.

        Args:
            test_unrolled_predictions (dict): Dictionary of unrolled
                predictions. Keys are "preds" with a list of predictions and
                label with whether or not it disrupted.
            high_threshold (float): High threshold for disruption.
            low_threshold (float): Low threshold for disruption.

        Returns:
            None.

        """

        thresholded_preds = []
        thresholded_labels = []

        for shot in test_unrolled_predictions:
            preds = test_unrolled_predictions[shot]["preds"]
            label = test_unrolled_predictions[shot]["label"]

            # initialize hysteresis counter
            hysteresis_counter = 0
            pred = 0
            for shot in range(len(preds)):
                for i in range(len(preds[shot])):
                    if preds[i] > high_threshold:
                        hysteresis_counter += 1
                    elif preds[i] < low_threshold:
                        hysteresis_counter = 0

                    if hysteresis_counter > hysteresis:
                        thresholded_preds.append(1)
                        break

                thresholded_preds.append(0)
                thresholded_labels.append(label)


'''





# -----------------------------------------------
def fpr_score(y_true, y_pred):
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return fp / (fp + tn)



# -----------------------------------------------
def tpr_score(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn)



# -----------------------------------------------
def compute_AUC_Zhu(unrolled_proba,logger):
    Disruptivity = []
    testTime_until_disrupt = []
    testClasses = []
    for _,item in unrolled_proba.items():
        Disruptivity.append(item['proba_shot'])
        testTime_until_disrupt.append(item['time_untill_disrupt'])
        testClasses.append(item['label_shot'])
    testClasses=np.array(testClasses)

    num_P=len(np.where(testClasses==1)[0]) # number of disruptive shots in test dataset
    num_N=len(np.where(testClasses==0)[0]) # number of non-disruptive shots in test dataset

    tpr_Zhu = []
    fpr_Zhu = []
    threshold_list = np.linspace(0,1, 1001)
    for i,t in enumerate(threshold_list):
        if i//100 == i/100:
            logger.info('AUC: loop over thresh. Done {} of {}'.format(i,len(threshold_list)))
        thresh = t
        num_test_sample = testClasses.shape[0]
        classes=np.zeros(num_test_sample,dtype='int16') # predicted class of the shot, if disruptive or not
        for i in range(num_test_sample):
            index=np.where(Disruptivity[i]>thresh)
            if len(index[0])>0:
                classes[i]=1 # if the predicted disruptivity > thresh -> label the shot as disruptive
                # ACHTUNG!!!: no check on where the disruptivity is > thresh, can happen at the beginning of the shot and then go to zero!!!

        indexx=np.where((classes==1)&(testClasses==1))[0] # shot indexes -> where the shot class is predicted correctly
        testTime_until_disrupt_effec=[testTime_until_disrupt[i] for i in indexx] # select time_until_disrupt for shots that are correctly labelled
        b=np.zeros(num_test_sample)  
        for i in range(num_test_sample): # loop over test shots
            index=np.where(Disruptivity[i]>thresh) # check where disruptivity is > thresh
            if len(index[0])>0: # if any instants where disruptivity > thresh 
                b[i]=index[0][0] # record the earliest time instant where disruptivity > thresh 
        b=b[indexx] # select time instants where the alarm is raised for truly disruptive shots
        b=np.asarray(b,dtype='int16') 
        tin=b#+10-1 # add 9 -> last time instant of the window
        time_new=[]
        for i in range(len(tin)): # loop over truly disruptive shots
            until=testTime_until_disrupt_effec[i].reshape(-1,1).tolist()
            time_new.append(until[tin[i]])
        time_new=np.asarray(time_new)
        tpr_Zhu.append(len(np.where(time_new>0.05)[0])/num_P) # label as true positive alarms raised at least 50ms in advance
        fpr_Zhu.append(len(np.where((classes==1)&(testClasses==0))[0])/num_N)
    fpr=np.array(fpr_Zhu)
    tpr=np.array(tpr_Zhu)
    auc_score_Zhu=auc(fpr, tpr)
    auc_score = auc_score_Zhu
    fpr_s = np.array(fpr)[np.argsort(fpr)]
    tpr_s = np.array(tpr)[np.argsort(fpr)]

    logger.info('AUC using Zhu\'s routines = {}'.format(auc_score))
    return auc_score





def auc(x, y):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.
    Parameters
    ----------
    x : ndarray of shape (n,)
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : ndarray of shape, (n,)
        y coordinates.
    Returns
    -------
    auc : float
    See Also
    --------
    roc_auc_score : Compute the area under the ROC curve.
    average_precision_score : Compute average precision from prediction scores.
    precision_recall_curve : Compute precision-recall pairs for different
        probability thresholds.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75
    """
    

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area






