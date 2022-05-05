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

################################### Base Model ##########################################
accumulated_acc = 0
test_accuracies10_Base = []
train_accuracies10 = []
test_epoch_accuracies10 = []
Losses10 = []
test_preds = []
test_exps = []
for i in range(0,10):
    begin = time.time()
    Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
        num_epochs, 
        learning_rate,
        WarpReg(train_ds.n_series,train_ds.n_labels,elasticity).to(device), 
        train_dl, 
        test_dl,
        scheduler= False, 
        decay=False,
        log=False)
    accumulated_acc += Acc
    test_accuracies10_Base.append(Acc)
    train_accuracies10.append(train_accuracies)
    test_epoch_accuracies10.append(test_accuracies)
    Losses10.append(Losses)
    test_preds.append(test_pred)
    test_exps.append(test_exp)
    end = time.time()
    print("ITERATION ",i,": ", (end - begin)/60)

acc_loss_10_plot(test_accuracies10_Base, train_accuracies10, test_epoch_accuracies10, Losses10, accumulated_acc/10, "WarpReg", data_name)
print("10 run Averaged Accuracy: ", accumulated_acc/10)
with open(result10_dir + "/" + data_name + "/WarpReg" + "/WarpReg.txt", 'a+') as f:
    for y_pred, y_exp in zip(test_preds, test_exps):
        f.write(classification_report(y_pred, y_exp))
    # Use if want to see 10 accuracies of base Model
    # f.write("test_accuracies10_Base:\n")
    # f.write(', '.join(str(e) for e in test_accuracies10_Base) + "\n")


################################### Compared Model ##########################################
accumulated_acc = 0
test_accuracies10 = []
train_accuracies10 = []
test_epoch_accuracies10 = []
Losses10 = []
test_preds = []
test_exps = []
for i in range(0,10):
    start = time.time()
    Acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp = standardCycle(
        num_epochs,  
        learning_rate,
        MockWarpReg(train_ds.n_series,train_ds.n_labels,elasticity).to(device), 
        train_dl, 
        test_dl,
        scheduler= False, 
        decay=False,
        log=False)

    accumulated_acc += Acc
    test_accuracies10.append(Acc)
    train_accuracies10.append(train_accuracies)
    test_epoch_accuracies10.append(test_accuracies)
    Losses10.append(Losses)
    test_preds.append(test_pred)
    test_exps.append(test_exp)
    end = time.time()
    print("ITERATION ",i,": ", (end - start)/60)

acc_loss_10_plot(test_accuracies10, train_accuracies10, test_epoch_accuracies10, Losses10, accumulated_acc/10, "MockWarpReg", data_name)
accuracy_record("MockWarpReg", data_name, test_preds, test_exps, test_accuracies10, test_accuracies10_Base)
print("10 run Averaged Accuracy: ", accumulated_acc/10)
