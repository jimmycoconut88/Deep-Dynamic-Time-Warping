from .lib import *
from .warp_function import compute_warp_matrix, compute_warp_path




def dtw_path_visualizer(dtw, path, title="Distance matrix"):
  plt.figure(figsize=(6, 4))
  plt.subplot(121)
  plt.title(title)
  plt.imshow(dtw, cmap=plt.cm.binary, interpolation="nearest", origin="bottom")
  x_path, y_path = zip(*path)
  plt.plot(y_path, x_path);

def dtw_map_visualizer(x,y, path):
  plt.figure()
  for x_i, y_j in path:
      plt.plot([x_i, y_j], [x[x_i] + 1.5, y[y_j] - 1.5], c="C7")
  plt.plot(np.arange(x.shape[0]), x + 1.5, "-o", c="C3")
  plt.plot(np.arange(y.shape[0]), y - 1.5, "-o", c="C0")
  plt.axis("off")
  plt.savefig("signals_a_b_align.pdf")

def loss_10_plot(accuracies, train_accuracies10, title, data):

  fig, axs = plt.subplots(10, figsize=(50,40))
  for idx, epochs in enumerate(train_accuracies10):
    axs[idx].plot(np.arange(len(epochs)), epochs, c="red", label="Training Loss")
    axs[idx].title.set_text("Accuracy: "+ str(accuracies[idx]))

  data_dir = result10_dir +"/"+ data +"/" + title
  if not os.path.exists(result10_dir):
    os.mkdir(result10_dir)
  if not os.path.exists(result10_dir +"/"+ data):
   os.mkdir(result10_dir +"/"+ data)
  if not os.path.exists(data_dir):
   os.mkdir(data_dir)
  plt.savefig(data_dir+"/Loss.png")

def acc_10_plot(train_accuracies, acc, title, data):


  fig, axs = plt.subplots(10, figsize=(50,40))
  for idx, epochs in enumerate(train_accuracies):
    axs[idx].plot(np.arange(len(epochs)), epochs, c="blue", label="Training Accuracies")
    axs[idx].legend()

  data_dir = result10_dir +"/"+ data +"/" + title
  if not os.path.exists(result10_dir):
    os.mkdir(result10_dir)
  if not os.path.exists(result10_dir +"/"+ data):
   os.mkdir(result10_dir +"/"+ data)
  if not os.path.exists(data_dir):
   os.mkdir(data_dir)
  plt.savefig(data_dir+"/Accuracy_"+str(round(acc,2))+".png")

def acc_loss_10_plot(test_accuracies, train_accuracies,test_epoch_accuracies, train_losses, avg_acc, title, data):

  fig1, axs1 = plt.subplots(10, figsize=(50,40))
  fig2, axs2 = plt.subplots(10, figsize=(50,40))

  for idx, (acc_train, acc_test, loss) in enumerate(zip(train_accuracies, test_epoch_accuracies, train_losses)):

    axs1[idx].title.set_text("Accuracy: "+ str(test_accuracies[idx]))
    axs2[idx].title.set_text("Accuracy: "+ str(test_accuracies[idx]))
    axs1[idx].plot(np.arange(len(acc_train)), acc_train, c="blue", label="Training Accuracies")
    axs1[idx].plot(np.arange(len(acc_test)), acc_test, c="orange", label="Testing Accuracies")
    axs1[idx].legend(loc=1)
    axs2[idx].plot(np.arange(len(loss)), loss, c="red", label="Training Loss")



  data_dir = result10_dir +"/"+ data +"/" + title
  if not os.path.exists(result10_dir):
    os.mkdir(result10_dir)
  if not os.path.exists(result10_dir +"/"+ data):
   os.mkdir(result10_dir +"/"+ data)
  if not os.path.exists(data_dir):
   os.mkdir(data_dir)
  fig1.savefig(data_dir+"/Accuracy"+str(round(avg_acc,2))+".png")
  fig2.savefig(data_dir+"/Loss"+str(round(avg_acc,2))+".png")


def acc_loss_plot(test_accuracies, train_accuracies,test_epoch_accuracies, train_losses, title, data):
    
  fig1, axs1 = plt.subplots(1)
  fig2, axs2 = plt.subplots(1)

  axs1.title.set_text("Accuracy: "+ str(test_accuracies))
  axs2.title.set_text("Accuracy: "+ str(test_accuracies))

  axs1.plot(np.arange(len(train_accuracies)), train_accuracies, c="blue", label="Training Accuracies")
  axs1.plot(np.arange(len(test_epoch_accuracies)), test_epoch_accuracies, c="orange", label="Testing Accuracies")

  axs1.legend(loc=1)
  axs2.plot(np.arange(len(train_losses)), train_losses, c="red", label="Training Loss")

  
  data_dir = result_dir +"/"+ data +"/" + title
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)
  if not os.path.exists(result_dir +"/"+ data):
   os.mkdir(result_dir +"/"+ data)
  if not os.path.exists(data_dir):
   os.mkdir(data_dir)
  fig1.savefig(data_dir+"/Accuracy_1_"+str(round(test_accuracies,2))+".png")
  fig2.savefig(data_dir+"/Loss_1_"+str(round(test_accuracies,2))+".png")


def warp(w, x):
  x_inflate = np.kron(x, np.ones((w.shape[0], 1), dtype=x.dtype))

  n_elasticity = w.shape[1]
  n_series = w.shape[0]
  # Score calculation
  scores = x_inflate*w
  warped_scores = compute_warp_matrix(np.array([scores]))

  warped_path = compute_warp_path(warped_scores)
  return warped_path, warped_scores

def data_plot(ds,tds, title):
  plt.figure()
  fig, ax = plt.subplots(nrows=2,gridspec_kw=dict(height_ratios=[1,1]), figsize=(50,30))
  labels = np.unique(ds.y)
  for x,label in zip(ds.X, ds.y):
    ax[0].plot(np.arange(x.shape[0]), x,linewidth=1.5 , c="C" + str(label),alpha=.5, label="Label" + str(label))

  for x,label in zip(tds.X, tds.y):
    ax[1].plot(np.arange(x.shape[0]), x,linewidth=1.5 , c="C" + str(label),alpha=.5, label="Label" + str(label))

  if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
  fig.savefig(plot_dir+"/"+title+".png")

def loss_plot(losses):
  ax = plt.gca()
  # ax.set_ylim([0, 0.05])
  ax.plot(np.arange(len(losses)), losses, c="b")
  plt.title("Loss graph over every epoch")
  plt.show()

def acc_plot(accuracies):
  ax = plt.gca()
  ax.plot(np.arange(len(accuracies)), accuracies, c="r")
  plt.title("Training accuracy over every epoch")
  plt.show()
def weight_heatmap(W):
  fig, ax = plt.subplots(nrows=W.shape[0],gridspec_kw=dict(height_ratios=[1]*W.shape[0]), figsize=(200,200))
  for idx, w in enumerate(W):
    sns.heatmap(W[idx], annot=False, ax=ax[idx], fmt='.2g')
  # plt.savefig("Weight 5 labels.png")
  # files.download("Weight 5 labels.png") 
  plt.show()


def weight_output_line_plot(weights,test, idx1, idx2, acc_reg, data_name, title, width=60, height = 25):
    labels = weights.shape[0]
    elasticities = weights.shape[1]
    
    fig, axs = plt.subplots(elasticities, labels * 2, figsize=(width,height))
    fig.suptitle('Output of weight and 2 test | Model Accuracy: {}'.format(round(acc_reg, 2)))
    series = []
    for label in range(labels):
        series.append( warp(weights[label],test.X[idx1])[1] )
        series.append( warp(weights[label],test.X[idx2])[1] )

    series = np.array(series)
    for col in range(labels*2):
        axs[0][col].title.set_text("Label: " + str(int(col/2)%labels) + " X_test: " + str(col%2))
        for row in range(elasticities):
            axs[row][col].plot(np.arange(series[col][0].shape[1]), series[col][0][row], c="C" + str(row))

    data_dir = plot_dir +"/"+ data_name
    if not os.path.exists(plot_dir):
      os.mkdir(plot_dir)
    if not os.path.exists(data_dir):
      os.mkdir(plot_dir +"/"+ data_name)
    fig.savefig(data_dir+"/{}_Output_lineplot_Label{}vs{}.png".format(title, test.y[idx1],test.y[idx2]))

def weight_output_heatmap(weights,test, idx1, idx2, acc_reg, data_name, title, width=100, height = 50):
    labels = weights.shape[0]
    fig, axs = plt.subplots(labels, 2, figsize=(width,height))
    fig.suptitle('Output of weight and 2 test | Model Accuracy: {}'.format(round(acc_reg, 2)))
    series = []
    for label in range(labels):
        series.append( warp(weights[label],test.X[idx1])[1] )
        series.append( warp(weights[label],test.X[idx2])[1] )
    series = np.array(series)
    for idx, seri in enumerate(series):
        sns.heatmap(seri[0],annot=False, linewidths=.5, ax=axs[int(idx/2)%labels][idx%2])

    data_dir = plot_dir +"/"+ data_name
    if not os.path.exists(plot_dir):
      os.mkdir(plot_dir)
    if not os.path.exists(data_dir):
      os.mkdir(plot_dir +"/"+ data_name)
    fig.savefig(data_dir+"/{}_Output_heatmap_Label{}vs{}.png".format(title, test.y[idx1],test.y[idx2]))

def weight_optimal_heatmap(weights, test_ds, acc_reg, data_name, title, width=100, height = 50):
    labels = weights.shape[0]
    elasticities = weights.shape[1]

    fig, axs = plt.subplots(labels, figsize=(width,height))
    fig.suptitle('Converging Optimal Path | Model Accuracy: {}'.format(round(acc_reg, 2)))
    for idx, label in enumerate(weights):
      optimal_path = np.zeros((elasticities, len(test_ds.X[0])))
      for data in test_ds.X:
        path, scores = warp(label,data)
        optimal_path+=path[0]
      sns.heatmap(optimal_path, annot=False, ax=axs[idx])
    data_dir = plot_dir +"/"+ data_name
    if not os.path.exists(plot_dir):
      os.mkdir(plot_dir)
    if not os.path.exists(data_dir):
      os.mkdir(plot_dir +"/"+ data_name)
    fig.savefig(data_dir+"/{}_Optimal_heatmap.png".format(title))


# Each row is a label from weight warp with dataset, Each column is an optimal path concatenated with dataset that has same label
def weight_label_optimal_heatmap(weights, test_ds, acc_reg, data_name, title, width=200, height = 150):
    labels = weights.shape[0]
    elasticities = weights.shape[1]

    fig, axs = plt.subplots(labels, labels, figsize=(width,height))
    fig.suptitle('Converging Optimal Path depend on label | Model Accuracy: {}'.format(round(acc_reg, 2)))
    for idx, label in enumerate(weights):
      optimal_path = np.zeros((labels, elasticities, len(test_ds.X[0])))
      for data in test_ds:
        path, scores = warp(label,data[0])
        optimal_path[data[1]]+=path[0]
      for jdx in range(labels):
        axs[idx][jdx].title.set_text("Weight for Label: " + str(idx) + " Correct Label: " + str(jdx))
        sns.heatmap(optimal_path[jdx], annot=False, ax=axs[idx][jdx]) #row, col
    data_dir = plot_dir +"/"+ data_name
    if not os.path.exists(plot_dir):
      os.mkdir(plot_dir)
    if not os.path.exists(data_dir):
      os.mkdir(plot_dir +"/"+ data_name)
    fig.savefig(data_dir+"/{}_Labels_Optimal_heatmap.png".format(title))

def accuracy_record(title, data_name, test_preds, test_exps, test_accuracies10, test_accuracies10_Base):
  with open(result10_dir + "/" + data_name + "/" + title + "/"+title +".txt", 'a+') as f:
    for y_pred, y_exp in zip(test_preds, test_exps):
      f.write(classification_report(y_pred, y_exp))

    f.write("test_accuracies10_Base:\n")
    f.write(', '.join(str(e) for e in test_accuracies10_Base) + "\n")
    f.write("test_accuracies10:\n")
    f.write(', '.join(str(e) for e in test_accuracies10) + "\n")

    w, p = wilcoxon(test_accuracies10, test_accuracies10_Base)
    f.write("Two-side:\n" + "w= "+ str(w) + "; " + "p= "+ str(p)  + "\n")
    w, p = wilcoxon(test_accuracies10, test_accuracies10_Base, alternative='greater')
    f.write("Greater:\n" + "w= "+ str(w) + "; " + "p= "+ str(p)  + "\n")
    w, p = wilcoxon(test_accuracies10, test_accuracies10_Base, alternative='less')
    f.write("Less:\n" + "w= "+ str(w) + "; " + "p= "+ str(p) + "\n")


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0.5):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True