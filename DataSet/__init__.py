from .CUB200 import CUB_200_2011
from .Car196 import Car196
from .Products import Products
from .WHALES import WHALES
from .In_shop_clothes import InShopClothes
# from .transforms import *
import os 

__factory = {
    'cub': CUB_200_2011,
    'car': Car196,
    'product': Products,
    'shop': InShopClothes,
    'whales':WHALES
}


def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        #this one is just hack for the whales to work, for the rest of dsets must use the commented out line
        root = root
       # root = os.path.join(root, get_full_name(name))
    
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)
