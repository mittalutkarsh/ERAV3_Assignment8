import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=2, bias=False,dilation=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )


        
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64,stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )
        
        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        )
        #self.pool1 = nn.MaxPool2d(2, 2)
        
        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )
        
        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
            
        )

        # TRANSITION BLOCK 2
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        )
        
        ###########--
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )
        
        # CONVOLUTION BLOCK 3
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )

        # TRANSITION BLOCK 3
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        )
        ##########

        ###########--
        self.convblock13 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )
        self.convblock14 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_value)
        )




        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.convblock15 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.convblock12(x)
        x = self.convblock13(x)
        x = self.convblock14(x)
            
        x = self.gap(x)
        x = self.convblock15(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
