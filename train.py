from common.lib import *
from common.data_processing import *
from common.utils import *
from common.ML import *
from common.FCN import *
from common.RESNET1D import *
###### Hyper-Parameter setting##################

# Shallow:
learning_rate = 0.01
# Deep:
learning_rate = 0.001
num_epochs = 2000
batch_size = 16
elasticity = 4
bias = -1

###### Data loading & pre-processing###########

# for data_name in ["Beef", "BirdChicken", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "ECG200", "GunPoint", "Ham", "ItalyPowerDemand", "Lightning2","Lightning7", "MiddlePhalanxTW"]:
data_name = "Beef"
device = get_default_device()
train_ds = UCRDataset(name=data_name,bias=bias)
test_ds = UCRDataset(name=data_name, test = True, bias = bias)

train_dl =DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)
#################################################### Traning Models #################################################

"""Architectures: Warp Softmax Regression & Mock Warp Softmax Regression"""

#################################################### BASE WSR #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    WarpReg(train_ds.n_series,train_ds.n_labels,elasticity).to(device), 
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "WSR", data_name)
#################################################### Deep WSR 1 (BatchNorm + Tanh) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    DWSR1(train_ds.n_labels, elasticity, train_ds.n_series).to(device),
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "DWSR1", data_name)
#################################################### Deep WSR 2 (BatchNorm + Tanh) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    DWSR2(train_ds.n_labels, elasticity, train_ds.n_series).to(device),
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "DWSR2", data_name)
#################################################### Deep WSR 3 (BatchNorm + Tanh) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    DWSR3(train_ds.n_labels, elasticity, train_ds.n_series).to(device),
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "DWSR3", data_name)
#################################################### Mock WSR #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs,  
    learning_rate,
    MockWarpReg(train_ds.n_series,train_ds.n_labels,elasticity).to(device), 
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "Mock WSR", data_name)
#################################################### Deep Mock WSR 1 (BatchNorm + Tanh) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs,  
    learning_rate,
    DMWSR1(train_ds.n_labels, elasticity, train_ds.n_series).to(device), 
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "DMWSR1", data_name)
#################################################### Deep Mock WSR 2 (BatchNorm + Tanh) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs,  
    learning_rate,
    DMWSR2(train_ds.n_labels, elasticity, train_ds.n_series).to(device), 
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "DMWSR2", data_name)
#################################################### Deep Mock WSR 3 (BatchNorm + Tanh) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs,  
    learning_rate,
    DMWSR3(train_ds.n_labels, elasticity, train_ds.n_series).to(device), 
    train_dl, 
    test_dl,
    scheduler= False, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "DMWSR3", data_name)
















"""Architectures: Fully Convolutional Network"""
#################################################### BASE FCN #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    FCN(train_ds.n_labels).to(device), 
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "FCN", data_name)
#################################################### Conv1D + FCN #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    FCCN(train_ds.n_labels).to(device), 
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "FCCN", data_name)
#################################################### Warp1D + FCN #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    WFCN(train_ds.n_labels, warped_elasticity = 4, warped_features = 3, warped_window= 3, warped_stride=1, maximize = True).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "WFCN", data_name)
#################################################### FWCN (Replace 1 Conv1D with Warp1D) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    FWCN(train_ds.n_labels, elasticity = 4).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "FWCN", data_name)
#################################################### FCN + WarpReg #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    FCNWSR(train_ds.n_labels, elasticity=4).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "FCNWSR", data_name)
#################################################### Warp1D + FCN + WarpReg #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    WFCNWSR(train_ds.n_labels, elasticity=4, warped_features = 3, warped_elasticity = 4, warped_window= 3, warped_stride=1, maximize = True).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "WFCNWSR", data_name)
#################################################### FWCNWSR (Replace First Conv1D with Warp1D, replace last Linear layer with WarpReg) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    FWCNWSR(train_ds.n_labels, elasticity=4).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "FWCNWSR", data_name)
#################################################### FWN (replace all Conv1D and Linear layer with Warp counterpart) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    FWN(train_ds.n_labels, elasticity=4).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False,
    log=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "FWCNWSR", data_name)











"""Architectures: Residual Network 1D"""
#################################################### BASE ResNet1D #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate,
    ResNet1D(block, num_classes = train_ds.n_labels).to(device), 
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "ResNet1D", data_name)
#################################################### Warp1D + ResNet1D #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate, 
    WResNet1D(block, num_classes = train_ds.n_labels, warped_elasticity = 4, warped_features = 3,  warped_window = 3, warped_stride = 1, maximize = True).to(device), 
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "WResNet1D", data_name)
#################################################### ResNet1D + WarpReg #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate, 
    ResNet1DWSR(block, num_classes = train_ds.n_labels, elasticity = 4).to(device), 
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "ResNet1DWSR", data_name)
#################################################### Warp1D + ResNet1D + WarpReg #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate, 
    WResNet1DWSR(block, num_classes = train_ds.n_labels, elasticity = 4, warped_elasticity= 4, warped_features = 3,  warped_window = 3, warped_stride = 1, maximize = True).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "WResNet1DWSR", data_name)
#################################################### ResWarpNet1D (Replace all instance of Conv1D and Linear layer with their Warp Counterpart) #################################################
Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
    num_epochs, 
    learning_rate, 
    ResWarpNet1D(warpblock, num_classes = train_ds.n_labels, elasticity = 4).to(device),
    train_dl, 
    test_dl,
    scheduler= True, 
    decay=False)
acc_loss_plot(Acc, train_accuracies, test_accuracies, Losses, "WResNet1DWSR", data_name)