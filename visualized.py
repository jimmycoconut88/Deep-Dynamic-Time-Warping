from common.lib import *
from common.data_processing import *
from common.utils import *
from common.ML import *
from common.FCN import *
from common.RESNET1D import *
###### Hyper-Parameter setting##################

# Shallow:
learning_rate = 0.01
num_epochs = 2000
batch_size = 16
elasticity = 4
bias = -1

###### Data loading & pre-processing###########

# for data_name in ["Beef", "BirdChicken", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "ECG200", "GunPoint", "Ham", "ItalyPowerDemand", "Lightning2","Lightning7", "MiddlePhalanxTW"]:
data_name = "Coffee"
device = get_default_device()
train_ds = UCRDataset(name=data_name,bias=bias)
test_ds = UCRDataset(name=data_name, test = True, bias = bias)

train_dl =DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


#################################################### WarpReg #################################################
warpreg = model_module(
    num_epochs, 
    learning_rate,
    WarpReg(train_ds.n_series,train_ds.n_labels,elasticity).to(device), 
    train_dl,
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
model_reg, acc_reg = warpreg
weight_reg = model_reg.weights.cpu().detach().numpy()
weight_output_line_plot(weight_reg, test_ds,0,27, acc_reg, data_name, title="WARPReg")
weight_output_heatmap(weight_reg, test_ds,0,27, acc_reg, data_name, title="WARPReg")
weight_optimal_heatmap(weight_reg, test_ds, acc_reg, data_name, title="WARPReg", width=100, height = 50)
weight_label_optimal_heatmap(weight_reg, test_ds, acc_reg, data_name, title="WARPReg",width=200, height = 50)
####################################################  Warp1D #################################################
warp1d = model_module(
    num_epochs, 
    learning_rate,
    Warp1D(warped_features=3, elasticity=4, kernel_size=3, stride=1).to(device), 
    train_dl,
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
model_1d, acc_1d = warp1d
weight_1d = model_1d.weights.cpu().detach().numpy()
weight_output_line_plot(weight_1d, test_ds,0,27, acc_1d, data_name, title="WARP1D")
weight_output_heatmap(weight_1d, test_ds,0,27, acc_1d, data_name, title="WARP1D")
weight_optimal_heatmap(weight_1d, test_ds, acc_1d, data_name, title="WARP1D", width=100, height = 50)
weight_label_optimal_heatmap(weight_1d, test_ds, acc_1d, data_name, title="WARP1D", width=200, height = 50)