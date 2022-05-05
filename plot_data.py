from common.lib import *
from common.data_processing import *
from common.utils import *
from common.ML import *
from common.FCN import *

for data_name in ["Beef", "BirdChicken", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "ECG200", "GunPoint", "Ham", "ItalyPowerDemand", "Lightning2","Lightning7", "MiddlePhalanxTW"]:
    train_ds = UCRDataset(name=data_name,bias=-1)
    test_ds = UCRDataset(name=data_name, test = True, bias = -1)
    data_plot(train_ds, test_ds, "data_name")

