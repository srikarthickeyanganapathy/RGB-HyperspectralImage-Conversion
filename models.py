import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            # FIXED: changed 0 to 1 (kernel_size=3, stride=1, padding=0)
            nn.Conv2d(dim, dim, 3, 1, 0), 
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            # FIXED: changed 0 to 1 (kernel_size=3, stride=1, padding=0)
            nn.Conv2d(dim, dim, 3, 1, 0), 
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=31, n_blocks=9, n_filters=64):
        super(ResNetGenerator, self).__init__()
        
        # 1. Initial Conv
        model = [nn.ReflectionPad2d(3), 
                 nn.Conv2d(input_nc, n_filters, kernel_size=7, padding=0), 
                 nn.InstanceNorm2d(n_filters), 
                 nn.ReLU(True)]

        # 2. Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(n_filters * mult, n_filters * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(n_filters * mult * 2), 
                      nn.ReLU(True)]

        # 3. ResNet Blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(n_filters * mult)]

        # 4. Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # ConvTranspose2d needs careful padding to match dimensions
            model += [nn.ConvTranspose2d(n_filters * mult, int(n_filters * mult / 2), 
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(n_filters * mult / 2)), 
                      nn.ReLU(True)]

        # 5. Output
        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(n_filters, output_nc, kernel_size=7, padding=0), 
                  nn.Tanh()]
                  
        self.model = nn.Sequential(*model)

    def forward(self, x): 
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=34): # 3 RGB + 31 HS
        super(NLayerDiscriminator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1), 
                 nn.LeakyReLU(0.2, True)]
                 
        model += [nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
                  nn.InstanceNorm2d(128), 
                  nn.LeakyReLU(0.2, True)]
                  
        model += [nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
                  nn.InstanceNorm2d(256), 
                  nn.LeakyReLU(0.2, True)]
                  
        model += [nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), 
                  nn.InstanceNorm2d(512), 
                  nn.LeakyReLU(0.2, True)]
                  
        model += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x): 
        return self.model(x)