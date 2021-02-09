import warnings
warnings.filterwarnings("ignore")
import h5py
from keras.models import load_model

from backgroundRemove import backgroundRemoval
from coloranalysis import get_colors
from DeepLearning import *


def ImageDetails(imagePath):
    image = backgroundRemoval(imagePath)
    get_colors(image,3,True)
    print("-"*20)
    print(f"Type of fashion: {FashionClassification(image)}")
    print(f"Brand: {LogoClassification(image)}")


