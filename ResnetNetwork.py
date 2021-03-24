from ResnetBasics import *


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels = 1, blocks_sizes=[16, 32, 64, 128, 256, 512], deepths=[2, 2, 2, 2, 2, 2],
                 activation='leaky_relu',block=ResNetBasicBlockEncoder, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        self.conv1 = nn.Conv1d(in_channels, self.blocks_sizes[0], kernel_size= 3, stride=1, padding=1, bias=False)
        self.batch = nn.BatchNorm1d(self.blocks_sizes[0])
        self.relu = activation_func(activation)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)


        self.block1 = ResNetBasicBlockEncoder(1,16,downsampling=2)
        self.block2 = ResNetBasicBlockEncoder(16,16)

        self.block3 = ResNetBasicBlockEncoder(16, 32)
        self.block4 = ResNetBasicBlockEncoder(32, 32)

        self.block5 = ResNetBasicBlockEncoder(32,64)
        self.block6 = ResNetBasicBlockEncoder(64,64)

        self.block7 = ResNetBasicBlockEncoder(64, 128)
        self.block8 = ResNetBasicBlockEncoder(128, 128)

        self.block9 = ResNetBasicBlockEncoder(128, 256)
        self.block10 = ResNetBasicBlockEncoder(256, 256)

        self.block11 = ResNetBasicBlockEncoder(256, 512, downsampling=2)
        self.block12 = ResNetBasicBlockEncoder(512, 512)

        self.block13 = ResNetBasicBlockEncoder(512, 1024)
        self.block14 = ResNetBasicBlockEncoder(1024, 1024)



        # self.gate = nn.Sequential(
        #     nn.Conv1d(in_channels, self.blocks_sizes[0], kernel_size= 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm1d(self.blocks_sizes[0]),
        #     activation(),
        #     nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayerEncoder(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayerEncoder(in_channels,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        # x = self.conv1(x.float())
        # x = self.batch(x)
        # x = self.relu(x)
        x = self.block1(x)
        #print(x.size())
        x = self.block2(x)
        #print(x.size())
        x = self.block3(x)
        #print(x.size())
        x = self.block4(x)
        #print(x.size())
        x = self.block5(x)
        #print(x.size())
        x = self.block6(x)
        #print(x.size())
        x = self.block7(x)
        #print(x.size())
        x = self.block8(x)
        #print(x.size())
        x = self.block9(x)
        #print(x.size())
        x = self.block10(x)
        #print(x.size())
        x = self.block11(x)
        #print(x.size())
        x = self.block12(x)
        #print(x.size())
        x = self.block13(x)
        # print(x.size())
        x = self.block14(x)

        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_channels=1, blocks_sizes=[128, 64, 32, 16], deepths=[2, 2, 2],
                 activation='leaky_relu', block=ResNetBasicBlockDecoder, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        self.last_block_index = len(blocks_sizes) - 1
        self.convT = nn.ConvTranspose1d(16, in_channels, kernel_size=3, padding=1, output_padding=0,bias=False)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.activation_function = activation_func(activation)

        self.block0 = ResNetBasicBlockDecoder(1024, 512)
        self.block00 = ResNetBasicBlockDecoder(512, 512)

        self.block1 = ResNetBasicBlockDecoder(512, 256, downsampling=2,output_padding= 1)
        self.block2 = ResNetBasicBlockDecoder(256, 256)

        self.block3 = ResNetBasicBlockDecoder(256, 128)
        self.block4 = ResNetBasicBlockDecoder(128, 128)

        self.block5 = ResNetBasicBlockDecoder(128, 64 ,downsampling=2,output_padding= 1)
        self.block6 = ResNetBasicBlockDecoder(64, 64)

        self.block7 = ResNetBasicBlockDecoder(64, 32)
        self.block8 = ResNetBasicBlockDecoder(32, 32)

        self.block9 = ResNetBasicBlockDecoder(32, 16, downsampling=2,output_padding= 1)
        self.block10 = ResNetBasicBlockDecoder(16, 16)



        # self.exit = nn.Sequential(
        #     nn.ConvTranspose1d(blocks_sizes[self.last_block_index], in_channels, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),
        #     nn.BatchNorm1d(in_channels),
        #     activation()
        # )
        #self.last_layer = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)

        self.in_out_block_sizes = list(zip(blocks_sizes[1:], blocks_sizes[2:]))
        self.blocks = nn.ModuleList([
            ResNetLayerDecoder(blocks_sizes[0], blocks_sizes[1], n=deepths[0], activation=activation,
                               block=block, *args, **kwargs),
            *[ResNetLayerDecoder(in_channels,
                                 out_channels, n=n, activation=activation,
                                 block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        # for block in self.blocks:
        #     x = block(x)
        # one_before_last = x.clone()
        # x = self.convT(x)
        x = self.block0(x)
        x = self.block00(x)
        x = self.block1(x)
        #print(x.size())
        x = self.block2(x)
        #print(x.size())
        x = self.block3(x)
        #print(x.size())
        x = self.block4(x)
        #print(x.size())
        x = self.block5(x)
        #print(x.size())
        x = self.block6(x)
        #print(x.size())
        x = self.block7(x)
        #print(x.size())
        x = self.block8(x)
        #print(x.size())
        x = self.block9(x)
        #print(x.size())
        x = self.block10(x)
        #print(x.size())
        x = self.convT(x)
        #print(x.size())
        return x