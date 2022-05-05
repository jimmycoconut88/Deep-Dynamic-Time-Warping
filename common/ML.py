from .lib import *
from .utils import reset_weights


def train(epoch, model, train_loader, optimizer, loss_function, log = False):
  # Set current loss value
  current_loss = 0.0
  total = 0
  # Iterate over the DataLoader for training data
  model.train()

  for batch_idx, (inputs, targets) in enumerate(train_loader,0): 

    # Zero the gradients
    optimizer.zero_grad()
    
    # Perform forward pass
    outputs = model(inputs)
    total+=outputs.shape[0]
    # Compute loss
    loss = loss_function(outputs, targets.long())
    # Perform backward pass
    loss.backward()
    
    # Perform optimization
    optimizer.step()

    # Print statistics
    current_loss += loss.detach().cpu().numpy()

    if log and batch_idx != 0 and batch_idx % (len(train_loader)-1) == 0: # For now the minibatch size is equal to a data size
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch,
            batch_idx+1,
            len(train_loader),
            100. * (batch_idx+1) / len(train_loader),
            current_loss/ total))
  return (current_loss/ total)

##########################################################################################
def test(epochs, model, test_loader, loss_function, log = False):
  model.eval()

  correct, test_loss = 0, 0
  total = 0
  y_pred_list = []
  y_exp_list = []
  with torch.no_grad():

    # Iterate over the test data and generate predictions

    for i, (inputs, targets) in enumerate(test_loader):

      # Generate outputs
      outputs = model(inputs)
        
      # Loss
      test_loss += loss_function(outputs, targets.long())
      # Set total and correct
      _, predicted = torch.max(outputs.data, 1)
      total+= len(predicted)
      correct += (predicted == targets.long()).sum().item()
      y_pred_list+=predicted.detach().cpu().numpy().tolist()
      y_exp_list+=targets.long().detach().cpu().numpy().tolist()
    # Print accuracy
    if log:
        print('Accuracy for epochs %d: %d %%' % (epochs, 100.0 * correct / total))
        print('--------------------------------')
    acc = 100.0 * (correct / total)
    loss  = test_loss / total 
  return acc, loss, y_pred_list, y_exp_list

#########################################################################################

def standardCycle(epochs,learning_rate, model, train_dl, test_dl,scheduler=True, decay = True, log = False):

  model.apply(reset_weights)
  train_accuracies = []
  test_accuracies = []
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  if decay == True:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-05)
  else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  # Learning rate reducer
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=0.0001)
  Losses = []
  for epoch in range(epochs):
    Losses.append(train(epoch, model, train_dl, optimizer, criterion, log))
    # Learning rate decay
    if scheduler == True:
      scheduler.step(Losses[-1])

    #See How accuracy improve over time
    train_acc, train_loss,train_y_pred, train_exp = test(epoch, model, train_dl, criterion, log)
    train_accuracies.append(train_acc)
    test_acc, test_loss,test_y_pred, test_exp = test(epoch, model, test_dl, criterion, log)
    test_accuracies.append(test_acc)

  test_acc, test_loss, test_pred, test_exp = test(epoch, model, test_dl, criterion, log)

  print(f"Accuracy on test set: {test_acc:.2f}")
  return test_acc, Losses, train_accuracies, test_accuracies, test_pred, test_exp

def model_module(epochs,learning_rate, model, train_dl, test_dl, scheduler=True, decay = True, log = False):
    
  model.apply(reset_weights)
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  if decay == True:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-05)
  else:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  # Learning rate reducer
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=0.0001)
  for epoch in range(epochs):
    loss = train(epoch, model, train_dl, optimizer, criterion, log)
    # Learning rate decay
    if scheduler == True:
      scheduler.step(loss)
  test_acc, test_loss, test_pred, test_exp = test(epoch, model, test_dl, criterion, log)
  return model, test_acc