from .lib import *
from .ESMR import *



class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()

        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.conv1 = nn.Conv1d(
            in_channels, 
            intermediate_channels, 
            kernel_size=8, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(intermediate_channels)

        self.conv2 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels)

        self.conv3 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(intermediate_channels)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        # print("OG shape: ", x.shape)
        x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0)) # Srly, how do u even pad even
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("Conv1 shape: ", x.shape)

        x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("Conv2 shape: ", x.shape)

        x = F.pad(x, (3 // 2 , 3 // 2,0,0))
        x = self.conv3(x)
        x = self.bn3(x)
        # print("Conv3 shape: ", x.shape)

        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            # print("Identity shape: ", identity.shape)
        x += identity
        x = self.relu(x)
        return x

class warpblock(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1, elasticity= 3
    ):
        super(warpblock, self).__init__()

        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.warp1 = Warp1D(intermediate_channels, elasticity=elasticity, kernel_size=8,stride=1)
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.warp2 = Warp1D(intermediate_channels, elasticity=elasticity, kernel_size=5,stride=1)
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.warp3 = Warp1D(intermediate_channels, elasticity=elasticity, kernel_size=3,stride=1)
        self.bn3 = nn.BatchNorm1d(intermediate_channels)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = F.pad(x, ( (8-1) // 2, (8) // 2 ,0,0)) # Srly, how do u even pad even
        x = self.warp1(x)
        x = self.bn1(x)

        x = F.pad(x, ( 5 // 2 , 5 // 2 ,0,0))
        x = self.warp2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = F.pad(x, (3 // 2 , 3 // 2,0,0))
        x = self.warp3(x)
        x = self.bn3(x)

        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

################################### RESNET1D ############################################
class ResNet1D(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet1D, self).__init__()
        self.in_channels = 1

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer = self._make_layer(
            block, intermediate_channels=64, stride=1
        )



        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # print("X FIRST SHAPE: ", x.shape)
        # rmb to remove after adding warp channelization
        x = torch.unsqueeze(x, 1)
        x = self.layer(x)
        # print("After feature extracting: ", x.shape)
        x = self.avgpool(x)
        # print("After AVG pooling: ", x.shape)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, intermediate_channels, stride):

        layers = []

        identity_downsample1 = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                intermediate_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels),
        )

        identity_downsample2 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels * 2),
        )

        identity_downsample3 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels * 2,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels  * 2),
        )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample1, stride)
        )
        layers.append(
            block(intermediate_channels, intermediate_channels * 2, identity_downsample2, stride)
        )
        layers.append(
            block(intermediate_channels * 2, intermediate_channels * 2, identity_downsample3, stride)
        )

        return nn.Sequential(*layers)

################################### RestNetESR ############################################
class ResNet1DWSR(nn.Module):
    def __init__(self, block, num_classes, elasticity = 4):
        super(ResNet1DWSR, self).__init__()
        self.in_channels = 1

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer = self._make_layer(
            block, intermediate_channels=64, stride=1
        )



        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = WarpReg(128, num_classes, elasticity)

    def forward(self, x):
        # print("X FIRST SHAPE: ", x.shape)
        # rmb to remove after adding warp channelization
        x = torch.unsqueeze(x, 1)
        x = self.layer(x)
        # print("After feature extracting: ", x.shape)
        x = self.avgpool(x)
        # print("After AVG pooling: ", x.shape)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, intermediate_channels, stride):
        layers = []

        identity_downsample1 = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                intermediate_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels),
        )

        identity_downsample2 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels * 2),
        )

        identity_downsample3 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels * 2,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels  * 2),
        )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample1, stride)
        )
        layers.append(
            block(intermediate_channels, intermediate_channels * 2, identity_downsample2, stride)
        )
        layers.append(
            block(intermediate_channels * 2, intermediate_channels * 2, identity_downsample3, stride)
        )

        return nn.Sequential(*layers)

################################### WResNet ############################################

class WResNet1D(nn.Module):
    def __init__(self, block, num_classes, warped_elasticity = 4, warped_features = 3,  warped_window = 3, warped_stride = 1, maximize = False):
        super(WResNet1D, self).__init__()
        if maximize == True:
          self.in_channels = 1
        else: self.in_channels = warped_features

        self.warp = Warp1D(warped_features, warped_elasticity, warped_window,warped_stride,maximize=maximize)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer = self._make_layer(
            block, intermediate_channels=64, stride=1
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # print("X FIRST SHAPE: ", x.shape)
        x = torch.unsqueeze(x, 1)
        x = self.warp(x)
        x = self.layer(x)
        # print("After feature extracting: ", x.shape)
        x = self.avgpool(x)
        # print("After AVG pooling: ", x.shape)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, intermediate_channels, stride):
        layers = []

        identity_downsample1 = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                intermediate_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels),
        )

        identity_downsample2 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels * 2),
        )

        identity_downsample3 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels * 2,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels  * 2),
        )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample1, stride)
        )
        layers.append(
            block(intermediate_channels, intermediate_channels * 2, identity_downsample2, stride)
        )
        layers.append(
            block(intermediate_channels * 2, intermediate_channels * 2, identity_downsample3, stride)
        )

        return nn.Sequential(*layers)

################################### WResNetESR ############################################
class WResNet1DWSR(nn.Module):
    def __init__(self, block, num_classes, elasticity = 4, warped_elasticity= 4, warped_features = 1,  warped_window = 3, warped_stride = 1, maximize = False):
        super(WResNet1DWSR, self).__init__()
        if maximize == True:
          self.in_channels = 1
        else: self.in_channels = warped_features

        self.warp = Warp1D(warped_features, warped_elasticity, warped_window,warped_stride,maximize=maximize)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer = self._make_layer(
            block, intermediate_channels=64, stride=1
        )



        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = WarpReg(128, num_classes, elasticity)

    def forward(self, x):
        # print("X FIRST SHAPE: ", x.shape)
        x = torch.unsqueeze(x, 1)
        x = self.warp(x)
        x = self.layer(x)
        # print("After feature extracting: ", x.shape)
        x = self.avgpool(x)
        # print("After AVG pooling: ", x.shape)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, intermediate_channels, stride):
        layers = []
        identity_downsample1 = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                intermediate_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels),
        )

        identity_downsample2 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels * 2),
        )

        identity_downsample3 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels * 2,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels  * 2),
        )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample1, stride)
        )
        layers.append(
            block(intermediate_channels, intermediate_channels * 2, identity_downsample2, stride)
        )
        layers.append(
            block(intermediate_channels * 2, intermediate_channels * 2, identity_downsample3, stride)
        )

        return nn.Sequential(*layers)

################################### ResWarpNet1D ############################################
class ResWarpNet1D(nn.Module):
    def __init__(self, block, num_classes, elasticity = 3):
        super(ResWarpNet1D, self).__init__()
        self.in_channels = 1

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer = self._make_layer(
            block, elasticity, intermediate_channels=64, stride=1
        )



        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = WarpReg(128, num_classes, elasticity)

    def forward(self, x):
        # print("X FIRST SHAPE: ", x.shape)
        # rmb to remove after adding warp channelization
        x = torch.unsqueeze(x, 1)
        x = self.layer(x)
        # print("After feature extracting: ", x.shape)
        x = self.avgpool(x)
        # print("After AVG pooling: ", x.shape)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, elasticity, intermediate_channels, stride):
        layers = []

        identity_downsample1 = nn.Sequential(
            Warp1D(intermediate_channels, elasticity=elasticity, kernel_size=1,stride=stride),
            nn.BatchNorm1d(intermediate_channels),
        )

        identity_downsample2 = nn.Sequential(
            Warp1D(intermediate_channels * 2, elasticity=elasticity, kernel_size=1,stride=stride),
            nn.BatchNorm1d(intermediate_channels * 2),
        )

        identity_downsample3 = nn.Sequential(
            Warp1D(intermediate_channels * 2, elasticity=elasticity, kernel_size=1,stride=stride), 
            nn.BatchNorm1d(intermediate_channels  * 2),
        )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample1, stride)
        )
        layers.append(
            block(intermediate_channels, intermediate_channels * 2, identity_downsample2, stride)
        )
        layers.append(
            block(intermediate_channels * 2, intermediate_channels * 2, identity_downsample3, stride)
        )

        return nn.Sequential(*layers)


################################### CResNet1D ############################################

class CResNet1D(nn.Module):
    def __init__(self, block, num_classes):
        super(CResNet1D, self).__init__()


        self.conv0 = nn.Conv1d(in_channels= 1, out_channels= 1, kernel_size=3,stride=1)
        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer = self._make_layer(
            block, intermediate_channels=64, stride=1
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # print("X FIRST SHAPE: ", x.shape)
        x = torch.unsqueeze(x, 1)
        x = self.conv0(x)
        x = self.layer(x)
        # print("After feature extracting: ", x.shape)
        x = self.avgpool(x)
        # print("After AVG pooling: ", x.shape)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, intermediate_channels, stride):
        layers = []

        identity_downsample1 = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                intermediate_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels),
        )

        identity_downsample2 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels * 2),
        )

        identity_downsample3 = nn.Sequential(
            nn.Conv1d(
                intermediate_channels * 2,
                intermediate_channels * 2,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm1d(intermediate_channels  * 2),
        )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample1, stride)
        )
        layers.append(
            block(intermediate_channels, intermediate_channels * 2, identity_downsample2, stride)
        )
        layers.append(
            block(intermediate_channels * 2, intermediate_channels * 2, identity_downsample3, stride)
        )

        return nn.Sequential(*layers)