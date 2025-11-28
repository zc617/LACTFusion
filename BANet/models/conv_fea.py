import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import os

class Conv3_Bn_LeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Bn_LeakyRelu2d, self).__init__()
        self.refpadding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        return self.lrelu(self.bn(self.conv(self.refpadding(x))))
    
class Conv3_Tanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Tanh2d, self).__init__()
        self.refpadding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv(self.refpadding(x)))/2 + 0.5
    
class Conv3_Bn_Relu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, dilation=1, groups=1):
        super(Conv3_Bn_Relu2d, self).__init__()
        self.ref_padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.ref_padding(x))))
     
class Feature_reconstruction(nn.Module):
    def __init__(self, dim):
        super(Feature_reconstruction, self).__init__()
        self.conv3_1 = Conv3_Bn_LeakyRelu2d(dim, dim // 2)  # 256-256
        self.conv3_2 = Conv3_Bn_LeakyRelu2d(dim // 2, dim // 4)  # 256-128
        self.conv3_3 = Conv3_Bn_LeakyRelu2d(dim // 4, dim // 8) #128 -64
        self.conv3_4 = Conv3_Bn_LeakyRelu2d(dim // 8, dim // 16) # 64-32
        self.conv3_5 = Conv3_Tanh2d(dim // 16, 1) #32 -1
        
    def forward(self,feature):
        conv3_1 = self.conv3_1(feature)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)
        conv3_5 = self.conv3_5(conv3_4)
        return conv3_5
    
# class Conv_block(nn.Module):
#     def __init__(self, outchannels):
#          super(Conv_block, self).__init__()
#          self.conv1 = Conv3_Bn_LeakyRelu2d(1, outchannels // 4)
#         #  self.conv2 = Conv3_Bn_LeakyRelu2d(outchannels // 8, outchannels // 4)
#          self.conv3 = Conv3_Bn_LeakyRelu2d(outchannels // 4, outchannels) 
#         #  self.conv4 =  nn.Conv2d(outchannels // 4, outchannels, 1, 1, 0)
#          self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         #  
#     def forward(self, x):
#          x1 = self.conv1(x)
#         #  x = self.conv2(x)  
#          x = self.conv3(x1)
#         #  x = self.lrelu(self.conv4(x1)) + x1
#          return x 

class Conv_block(nn.Module):
      def __init__(self, outchannels):
         super(Conv_block, self).__init__()
         self.conv2 = Conv3_Bn_LeakyRelu2d(1, outchannels // 8) # 128 / 8
         self.conv3 = Conv3_Bn_LeakyRelu2d(outchannels // 8, outchannels // 8 ) 
         self.conv4 = Conv3_Bn_LeakyRelu2d(outchannels // 4, outchannels // 8 ) 
         self.conv5 = nn.Conv2d((outchannels  // 8) * 3, outchannels, 1, 1, 0)
         self.conv6 = nn.Conv2d(outchannels // 8, outchannels, 1, 1, 0)
         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
         self.tanh = nn.Tanh()
        #  
      def forward(self, x):
         x1 = self.conv2(x) # 1-16 
         x2 = self.conv3(x1)
         x3 = self.conv4(torch.cat((x1, x2), 1))  
         x4 = self.lrelu(self.conv5(torch.cat((x2, torch.cat((x3, x2), 1)), 1)))
         x5 = self.lrelu(self.conv6(x1))
         return x5 + x4
