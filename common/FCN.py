from .lib import *
from .ESMR import *


class FCN(nn.Module):
    def __init__(self, n_labels):
        super(FCN, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels= 1, out_channels= 128, kernel_size=8,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels= 256, out_channels= 128, kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )


        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
            nn.Linear(128, n_labels)
        )
    def forward(self, x):
      x = F.pad(x.unsqueeze(1), ( (8-1) // 2, (8) // 2 ,0,0))
      x = self.conv1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.conv2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.conv3(x)
      x = self.pool(x)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x




###################### FCCN #####################################


class FCCN(nn.Module):
    def __init__(self, n_labels):
        super(FCCN, self).__init__()

        self.conv0 = nn.Conv1d(in_channels= 1, out_channels= 1, kernel_size=3,stride=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels= 1, out_channels= 128, kernel_size=8,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels= 256, out_channels= 128, kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )


        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
            nn.Linear(128, n_labels)
        )
    def forward(self, x):

      x = self.conv0(x.unsqueeze(1))
      x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0))
      x = self.conv1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.conv2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.conv3(x)
      # print("AFTER FCN: ",x.shape)
      x = self.pool(x)
      # print("AFTER POOL: ", x.shape)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x

###################### WFCN #####################################

class WFCN(nn.Module):
    def __init__(self, 
                 n_labels, warped_elasticity = 4, warped_features = 3, warped_window = 3, warped_stride=1,
                 maximize = False,
                 w = None):
        super(WFCN, self).__init__()

        self.warp = Warp1D(warped_features, warped_elasticity, warped_window,warped_stride,maximize=maximize, w = w)

        if maximize == True:
          warped_features = 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels= 1, out_channels= 128, kernel_size=8,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels= 256, out_channels= 128, kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )


        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
            nn.Linear(128, n_labels)

        )
    def forward(self, x):
      x = torch.unsqueeze(x, 1)
      x = self.warp(x)
      x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0))
      x = self.conv1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.conv2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.conv3(x)
      x = self.pool(x)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x

###################### FCNESR #####################################
class FCNWSR(nn.Module):
    def __init__(self, 
                 n_labels, elasticity,
                 w = None):
        super(FCNWSR, self).__init__()



        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels= 1, out_channels= 128, kernel_size=8,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels= 256, out_channels= 128, kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
          WarpReg(128, n_labels, elasticity, w = w),
        )
    def forward(self, x):
      x = F.pad(x.unsqueeze(1), ( (8-1) // 2, (8) // 2 ,0,0))
      x = self.conv1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.conv2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.conv3(x)
      x = self.pool(x)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x
    
###################### WFCNWSR #####################################
class WFCNWSR(nn.Module):
    def __init__(self, 
                 n_labels, elasticity, warped_features,
                 warped_elasticity = 4,warped_window = 3, warped_stride=1,
                 maximize = False, w = None):
        super(WFCNWSR, self).__init__()

        self.warp = Warp1D(warped_features, warped_elasticity, warped_window,warped_stride,maximize=maximize, w = w)
        
        if maximize == True:
          warped_features = 1

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels= 1, out_channels= 128, kernel_size=8,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels= 256, out_channels= 128, kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
          WarpReg(128, n_labels, elasticity, w = w),
        )
    def forward(self, x):
      x = torch.unsqueeze(x, 1)
      x = self.warp(x)
      x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0))
      x = self.conv1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.conv2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.conv3(x)
      x = self.pool(x)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x


###################### FWN #####################################
class FWN(nn.Module):
    def __init__(self, n_labels, elasticity):
        super(FWN, self).__init__()


        self.warp1 = nn.Sequential(
            Warp1D(warped_features = 128, elasticity= elasticity, kernel_size = 8, stride= 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.warp2 = nn.Sequential(
            Warp1D(warped_features = 256, elasticity=elasticity, kernel_size = 5, stride= 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.warp3 = nn.Sequential(
            Warp1D(warped_features = 128, elasticity=elasticity, kernel_size = 3, stride= 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
          WarpReg(128, n_labels, elasticity),
          
        )
    def forward(self, x):
      x = torch.unsqueeze(x, 1)
      x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0)) 
      x = self.warp1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.warp2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.warp3(x)
      x = self.pool(x)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x



###################### FWCN #####################################


class FWCN(nn.Module):
    def __init__(self, n_labels, elasticity):
        super(FWCN, self).__init__()


        self.warp1 = nn.Sequential(
            Warp1D(warped_features = 128, elasticity= elasticity, kernel_size = 8, stride= 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels= 256, out_channels= 128, kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )


        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
            nn.Linear(128, n_labels)
        )
    def forward(self, x):
      x = torch.unsqueeze(x, 1)
      x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0))
      x = self.warp1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.conv2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.conv3(x)
      x = self.pool(x)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x


###################### FWCNESR #####################################


class FWCNWSR(nn.Module):
    def __init__(self, n_labels, elasticity):
        super(FWCNWSR, self).__init__()


        self.warp1 = nn.Sequential(
            Warp1D(warped_features = 128, elasticity= elasticity, kernel_size = 8, stride= 1),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels= 128, out_channels= 256, kernel_size=5,stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels= 256, out_channels= 128, kernel_size=3,stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            )


        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reg = nn.Sequential(
            WarpReg(128, n_labels, elasticity),
        )
    def forward(self, x):
      x = torch.unsqueeze(x, 1)
      x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0))
      x = self.warp1(x)
      x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
      x= self.conv2(x)
      x = F.pad(x, ( 3 // 2 , 3 // 2 ,0,0))
      x= self.conv3(x)
      x = self.pool(x)
      x = x.flatten(start_dim= 1)
      x = self.reg(x)
      return x
