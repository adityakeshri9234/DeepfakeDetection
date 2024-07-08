import torch
import torch.nn as nn
import torch.nn.functional as F

# MBConv Block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expansion_factor, stride=1):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * expansion_factor

        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(mid_channels)
        self.dw_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu6(self.bn0(self.expand_conv(x)))
        out = F.relu6(self.bn1(self.dw_conv(out)))
        out = self.bn2(self.project_conv(out))
        if self.use_residual:
            out = x + out
        return out

# Upper Branch
class UpperBranch(nn.Module):
    def __init__(self):
        super(UpperBranch, self).__init__()
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        
        self.mbconv1 = nn.Sequential(
            MBConvBlock(32, 16, kernel_size=3, expansion_factor=1, stride=1),
            MBConvBlock(16, 16, kernel_size=3, expansion_factor=1, stride=1)
        )
        
        self.mbconv2 = nn.Sequential(
            MBConvBlock(16, 24, kernel_size=3, expansion_factor=6, stride=2),
            MBConvBlock(24, 24, kernel_size=3, expansion_factor=6, stride=1),
            MBConvBlock(24, 24, kernel_size=3, expansion_factor=6, stride=1),
            MBConvBlock(24, 24, kernel_size=3, expansion_factor=6, stride=1)
        )
        
        self.mbconv3 = nn.Sequential(
            MBConvBlock(24, 40, kernel_size=5, expansion_factor=6, stride=2),
            MBConvBlock(40, 40, kernel_size=5, expansion_factor=6, stride=1),
            MBConvBlock(40, 40, kernel_size=5, expansion_factor=6, stride=1),
            MBConvBlock(40, 40, kernel_size=5, expansion_factor=6, stride=1)
        )
        
        self.mbconv4 = nn.Sequential(
            MBConvBlock(40, 80, kernel_size=3, expansion_factor=6, stride=2),
            MBConvBlock(80, 80, kernel_size=3, expansion_factor=6, stride=1),
            MBConvBlock(80, 80, kernel_size=3, expansion_factor=6, stride=1),
            MBConvBlock(80, 80, kernel_size=3, expansion_factor=6, stride=1),
            MBConvBlock(80, 80, kernel_size=3, expansion_factor=6, stride=1),
            MBConvBlock(80, 80, kernel_size=3, expansion_factor=6, stride=1)
        )
        
        self.mbconv5 = nn.Sequential(
            MBConvBlock(80, 112, kernel_size=5, expansion_factor=6, stride=1),
            MBConvBlock(112, 112, kernel_size=5, expansion_factor=6, stride=1),
            MBConvBlock(112, 112, kernel_size=5, expansion_factor=6, stride=1),
            MBConvBlock(112, 112, kernel_size=5, expansion_factor=6, stride=1),
            MBConvBlock(112, 112, kernel_size=5, expansion_factor=6, stride=1),
            MBConvBlock(112, 112, kernel_size=5, expansion_factor=6, stride=1)
        )
        
        self.mbconv6 = nn.Sequential(
            MBConvBlock(112, 160, kernel_size=3, expansion_factor=6, stride=2),
            MBConvBlock(160, 160, kernel_size=3, expansion_factor=6, stride=1)
        )

    def forward(self, x):
        x = F.relu6(self.bn(self.initial_conv(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        return x

# Patch Embedding Layer
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * patch_size * 3, embed_dim)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size * 3)
        embedded_patches = self.projection(patches)
        return embedded_patches

# SAM Block
class SAMBlock(nn.Module):
    def __init__(self, in_channels):
        super(SAMBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, padding=0, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        query = query.permute(0, 2, 1)
        attention = torch.matmul(query, key)
        attention = F.softmax(attention / (channels ** 0.5), dim=-1)

        out = torch.matmul(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma * out + x

        return out

# SAM Module
class SAMModule(nn.Module):
    def __init__(self, in_channels):
        super(SAMModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.sam_block = SAMBlock(in_channels)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.sam_block(x)
        return x

# Lower Branch
class LowerBranch(nn.Module):
    def __init__(self, input_shape, patch_size, embed_dim):
        super(LowerBranch, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        self.sam_modules = nn.Sequential(
            SAMModule(embed_dim),
            SAMModule(embed_dim),
            SAMModule(embed_dim)
        )
        self.conv1x1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.patch_embedding(x)
        num_patches_sqrt = self.input_shape[0] // self.patch_size
        x = x.view(-1, self.embed_dim, num_patches_sqrt, num_patches_sqrt)
        x = self.sam_modules(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Full Model
class SADFFDModel(nn.Module):
    def __init__(self, input_shape):
        super(SADFFDModel, self).__init__()
        self.upper_branch = UpperBranch()
        self.lower_branch = LowerBranch(input_shape, patch_size=16, embed_dim=160)  # Match the upper branch's final out_channels

        self.conv1x1 = nn.Conv2d(320, 128, kernel_size=1, padding=0, bias=False)  # 160 (Upper) + 160 (Lower) = 320
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        upper_output = self.upper_branch(x)
        lower_output = self.lower_branch(x)

        # Resize lower_output to match upper_output's spatial dimensions
        lower_output = F.interpolate(lower_output, size=upper_output.shape[2:4], mode='bilinear', align_corners=False)

       # print("Shape of upper_output:", upper_output.shape)  # Print the shape of upper_output
        #print("Shape of lower_output:", lower_output.shape)  # Print the shape of lower_output

        concatenated = torch.cat((upper_output, lower_output), dim=1)
        x = self.conv1x1(concatenated)
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

# Example Usage
input_shape = (224, 224, 3)  # Example input shape
model = SADFFDModel(input_shape)

# Testing the model with dummy input
x = torch.randn(1, 3, 224, 224)  # Batch size of 1
output = model(x)
print(output)
