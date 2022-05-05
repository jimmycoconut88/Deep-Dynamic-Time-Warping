from .lib import *

class UCRDataset(Dataset):
  def __init__(self, name = "Beef",test = False, bias = 0):
    # Data Loading
    if test:
      data_type = "_TEST.tsv"
    else:
      data_type = "_TRAIN.tsv"
    Xy = np.loadtxt(open(data_dir + "/" + name + "/" + name + data_type, "rb"), delimiter="\t", skiprows=0,dtype=np.float32)
    self.X = torch.from_numpy(Xy[:, 1:])

    self.y = np.unique(torch.from_numpy(Xy[:, 0]), return_inverse=True)[1]
    self.n_samples = Xy.shape[0]
    self.add_bias(bias)
    self.n_series = self.X.shape[1]
    self.n_labels = len({item[0] for item in Xy})
    
  def add_bias(self, bias):
    biases = np.full((self.n_samples, 1), bias,dtype=np.float32)
    self.X = np.concatenate((biases, self.X), axis = 1)
  def __getitem__(self, index):
    return self.X[index],self.y[index]
  def __len__(self):
    return self.n_samples

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
