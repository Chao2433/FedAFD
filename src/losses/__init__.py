from .triplet import TripletLoss,TripletLoss_uni
from .CenterTriplet import CenterTripletLoss
from .GaussianMetric import GaussianMetricLoss
from .HistogramLoss import HistogramLoss
from .BatchAll import BatchAllLoss
from .NeighbourLoss import NeighbourLoss
from .DistanceMatchLoss import DistanceMatchLoss
from .NeighbourHardLoss import NeighbourHardLoss
from .OLE import OLELoss,OLELoss_uni
import torch.nn as nn

__factory = {
    'triplet': TripletLoss,
    'triplet_uni':TripletLoss_uni,
    'histogram': HistogramLoss,
    'gaussian': GaussianMetricLoss,
    'batchall': BatchAllLoss,
    'neighbour': NeighbourLoss,
    'distance_match': DistanceMatchLoss,
    'neighard': NeighbourHardLoss,
    'softmax':nn.CrossEntropyLoss,
    'OLEloss':OLELoss,
    'OLEloss_uni':OLELoss_uni,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]( *args, **kwargs)
