from abc import ABCMeta, abstractmethod
import torch
import numpy as np

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        ## TODO implement
        pass



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

       ##TODO implement
        pass
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        ##TODO implement
        pass
          
    def compute_iou(y_true, y_pred):
        intersection = np.sum(np.logical_and(y_true, y_pred))
        union = np.sum(np.logical_or(y_true, y_pred))
        return intersection / union


    
    def mIoU(self,y_true, y_pred, num_classes) -> float: #  create obj and send the param n remove from here

        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        
        miou_total = 0.0
        for i in range(1, num_classes + 1):  # Assuming class labels start from 1
            y_true_class = (y_true == i)
            y_pred_class = (y_pred == i)
            miou_class = compute_iou(y_true_class, y_pred_class)
            miou_total += miou_class
        return miou_total / num_classes
        





