from .lib import *
from .warp_function import *

##################################### Warp Function
class _WarpFunction(Function):
  @staticmethod
  def forward(ctx, scores):
    dev = scores.device
    dtype = scores.dtype
    scores_ = scores.detach().cpu().numpy()
    warped_masks = torch.Tensor().to(dev).type(dtype)
    warped_matrices = torch.Tensor().to(dev).type(dtype)
    for batch in scores_:
      warped_matrices =  torch.cat((warped_matrices,torch.Tensor(np.array([compute_warp_matrix(batch)],dtype=np.float32)).to(dev).type(dtype)), 0).to(dev).type(dtype)
      calc_mask = torch.Tensor([compute_warp_path(warped_matrices[-1].cpu().numpy())]).to(dev).type(dtype)

      warped_masks = torch.cat((warped_masks, calc_mask), 0)
    ctx.save_for_backward(warped_masks, scores)
    return warped_matrices[:,:,-1,-1]

  @staticmethod
  def backward(ctx, grad_output):
    warped_masks,  scores = ctx.saved_tensors
    dev = warped_masks.device
    warped_masks = warped_masks.detach().cpu().numpy()
    grad_output = grad_output.detach().cpu().numpy()
    return torch.tensor(np.einsum("bcen,bc->bcen",warped_masks,grad_output)).to(dev), None


class _Warp1DFunction(Function):
  @staticmethod
  def forward(ctx, scores,maximize=False):
    dev = scores.device
    dtype = scores.dtype
    scores_ = scores.detach().cpu().numpy()
    warped_masks = torch.Tensor().to(dev).type(dtype)
    warped_matrices = torch.Tensor().to(dev).type(dtype)
    for batch in scores_:
      warped_matrices =  torch.cat((warped_matrices,torch.Tensor(np.array([compute_warp_feature(batch)],dtype=np.float32)).to(dev).type(dtype)), 0).to(dev).type(dtype)
      calc_mask = torch.Tensor([compute_warp_feature_path(warped_matrices[-1].cpu().numpy())]).to(dev).type(dtype)
      warped_masks = torch.cat((warped_masks, calc_mask), 0)

    if maximize == True:
      # maximized output
      out = warped_matrices[:,:,:,-1,-1]                # grab return value
      sum = torch.sum(out,2)                            # Sum each channels up
      max_idx = torch.argmax(sum, dim=1)                # Select summed channel with max value 
      out = out[torch.arange(out.size(0)), max_idx]     # only take value of that 1 channel
      out = torch.unsqueeze(out, 1)                     # Unsqueeze to craete channel dim

      # maximized gradient
      max_mask = torch.zeros(warped_masks.shape).to(dev).type(dtype)    # Create new mask with same dim
      max_mask[torch.arange(warped_masks.size(0)),max_idx, : ,:,:] = warped_masks[torch.arange(warped_masks.size(0)),max_idx, :,:,:]    # only apply mask for the chosen channel to gradient the chosen mask
      warped_mask = max_mask
      ctx.save_for_backward(max_mask)
      return out

    ctx.save_for_backward(warped_masks)
    return warped_matrices[:,:,:,-1,-1]

  @staticmethod
  def backward(ctx, grad_output):
    warped_masks, = ctx.saved_tensors
    dev = warped_masks.device
    warped_masks = warped_masks.detach().cpu().numpy()
    grad_output = grad_output.detach().cpu().numpy()
    return torch.tensor(np.einsum("bfcen,bfc->bfcen",warped_masks,grad_output)).to(dev), None

class _MockWarpFunction(Function):
  @staticmethod
  def forward(ctx, scores):
    dev = scores.device
    dtype = scores.dtype
    sum = torch.sum(scores,3)                            # Sum each channels up  
    max = torch.max(sum,dim=2)
    #mask
    max_mask = torch.zeros(scores.shape).to(dev).type(dtype)
    max_mask = torch.flatten(max_mask,0,1)
    max_mask[torch.arange(max_mask.size(0)),max.indices.view(-1)] = 1
    max_mask = torch.reshape(max_mask, scores.shape)
    ctx.save_for_backward(max_mask)
    return max.values

########################################################### Warp module
  @staticmethod
  def backward(ctx, grad_output):
    max_mask, = ctx.saved_tensors
    return torch.einsum("bcen,bc->bcen",max_mask,grad_output), None


class Warp1D(nn.Module):
    def __init__(self, warped_features, elasticity, kernel_size, stride, maximize = False, w = None):
      super(Warp1D, self).__init__()
      self.warped_features = warped_features
      self.elasticity = elasticity
      self.kernel_size = kernel_size
      self.stride = stride
      self.maximize = maximize


      # For testing
      if w != None:
        self.weights = nn.Parameter(w, requires_grad = True)
      else:
        weights = torch.randn(warped_features, elasticity, kernel_size, dtype=torch.float32)
        self.weights = nn.Parameter(weights, requires_grad = True)
        # Kaiming for ReLu
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

      self.func_warp = _Warp1DFunction.apply

    
    def scores(self, x, w):
      # w: label, elastic, n_series
      # x: batch, channel_split, n_series
      return torch.einsum("fen,bcn->bfcen",w,x)

    def naiveConv(self, x, size, stride):
      x = torch.einsum("bin->bn", x)
      if (x.shape[1] - size) % stride != 0:
        raise RuntimeError(f"(input length - kernel size) must be divisible by stride: input length is {x.shape[1]}") 
      #n_channel = ((input length - kernel) / stride)  + 1
      return x.unfold(1,size,stride)

    def forward(self, x):
      # Split it into mini-series to mult with w
      x = self.naiveConv(x,self.kernel_size, self.stride)
      score_xw = self.scores(x, self.weights)
      out_xw = self.func_warp(score_xw, self.maximize)
      return out_xw


class WarpReg(nn.Module):
    def __init__(self, input, output, elasticity, w = None):
      super(WarpReg, self).__init__()

      self.input = input
      self.output = output
      self.elasticity = elasticity

      # For testing
      if w != None:
        self.weights = nn.Parameter(w, requires_grad = True)
      else:

        weights = torch.randn(output,elasticity, input, dtype=torch.float32)
        self.weights = nn.Parameter(weights, requires_grad = True)
        nn.init.normal_(self.weights,mean=0,std=1.0)

      self.func_warp = _WarpFunction.apply

    
    def scores(self, x, w):

      return torch.einsum("ijk,bk->bijk",w,x)

    def forward(self, x):
      score_xw = self.scores(x, self.weights)
      out_xw = self.func_warp(score_xw)
      return out_xw

class MockWarpReg(nn.Module):
    def __init__(self, input, output, elasticity, w = None):
      super(MockWarpReg, self).__init__()

      self.input = input
      self.output = output
      self.elasticity = elasticity

      # For testing
      if w != None:
        self.weights = nn.Parameter(w, requires_grad = True)
      else:

        weights = torch.randn(output,elasticity, input, dtype=torch.float32)
        self.weights = nn.Parameter(weights, requires_grad = True)
        nn.init.normal_(self.weights,mean=0,std=1.0)

      self.func_warp = _MockWarpFunction.apply

    
    def scores(self, x, w):

      return torch.einsum("ijk,bk->bijk",w,x)

    def forward(self, x):
      score_xw = self.scores(x, self.weights)
      out_xw = self.func_warp(score_xw)
      return out_xw

############################## Deep Warp Module
class DWSR1(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DWSR1, self).__init__()

        self.dense = nn.Sequential(
            WarpReg(int(n_series), n_labels, elasticity, w = w),
            nn.BatchNorm1d(num_features=n_labels),
            nn.Tanh(),
        )
    def forward(self, x):
      x = self.dense(x) #
      return x

class DWSR2(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DWSR2, self).__init__()

        self.dense = nn.Sequential(
            WarpReg(int(n_series), 50, elasticity, w = w),
            nn.BatchNorm1d(num_features=50),
            nn.Tanh(),
            WarpReg(50, n_labels, elasticity, w = w),
            nn.BatchNorm1d(num_features=n_labels),
            nn.Tanh(),
        )
    def forward(self, x):
      x = self.dense(x) #
      return x
  
class DWSR3(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DWSR3, self).__init__()

        self.dense = nn.Sequential(
            WarpReg(int(n_series), 100, elasticity, w = w),
            nn.BatchNorm1d(num_features=100),
            nn.Tanh(),
            WarpReg(100, 50, elasticity, w = w),
            nn.BatchNorm1d(num_features=50),
            nn.Tanh(),
            WarpReg(50, n_labels, elasticity, w = w),
            nn.BatchNorm1d(num_features=n_labels),
            nn.Tanh(),
        )
    def forward(self, x):
      x = self.dense(x) #
      return x

##############################DEEP WARP MODULE LEAKY RELU
############################## Deep Warp Module
class DLRWSR1(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DLRWSR1, self).__init__()

        self.dense = nn.Sequential(
            WarpReg(int(n_series), n_labels, elasticity, w = w),
            nn.LeakyReLU()
        )
    def forward(self, x):
      x = self.dense(x) #
      return x

class DLRWSR2(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DLRWSR2, self).__init__()

        self.dense = nn.Sequential(
            WarpReg(int(n_series), 50, elasticity, w = w),
            nn.LeakyReLU(),
            WarpReg(50, n_labels, elasticity, w = w),
            nn.LeakyReLU()
        )
    def forward(self, x):
      x = self.dense(x) #
      return x
  
class DLRWSR3(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DLRWSR3, self).__init__()

        self.dense = nn.Sequential(
            WarpReg(int(n_series), 100, elasticity, w = w),
            nn.LeakyReLU(),
            WarpReg(100, 50, elasticity, w = w),
            nn.LeakyReLU(),
            WarpReg(50, n_labels, elasticity, w = w),
            nn.LeakyReLU()
        )
    def forward(self, x):
      x = self.dense(x) #
      return x

############################## Deep mock Warp Module
class DMWSR1(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DMWSR1, self).__init__()

        self.dense = nn.Sequential(
            MockWarpReg(int(n_series), n_labels, elasticity, w = w),
            nn.BatchNorm1d(num_features=n_labels),
            nn.Tanh(),
        )
    def forward(self, x):
      x = self.dense(x) #
      return x

class DMWSR2(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DMWSR2, self).__init__()

        self.dense = nn.Sequential(
            MockWarpReg(int(n_series), 50, elasticity, w = w),
            nn.BatchNorm1d(num_features=50),
            nn.Tanh(),
            MockWarpReg(50, n_labels, elasticity, w = w),
            nn.BatchNorm1d(num_features=n_labels),
            nn.Tanh(),
        )
    def forward(self, x):
      x = self.dense(x) #
      return x
  
class DMWSR3(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, n_series,
                 w = None):
        super(DMWSR3, self).__init__()

        self.dense = nn.Sequential(
            MockWarpReg(int(n_series), 100, elasticity, w = w),
            nn.BatchNorm1d(num_features=100),
            nn.Tanh(),
            MockWarpReg(100, 50, elasticity, w = w),
            nn.BatchNorm1d(num_features=50),
            nn.Tanh(),
            MockWarpReg(50, n_labels, elasticity, w = w),
            nn.BatchNorm1d(num_features=n_labels),
            nn.Tanh(),
        )
    def forward(self, x):
      x = self.dense(x) #
      return x