# -*- coding: utf-8 -*-

from fcnn import metrics
from fcnn.schedulers import SchedulerType
from fcnn.trainer import Trainer
from fcnn.fcnn import build_model, finalize_model

__all__ = [build_model,
           finalize_model,
           SchedulerType,
           Trainer]


custom_objects = {'mean_iou': metrics.mean_iou,
                  'dice_coefficient': metrics.dice_coefficient}