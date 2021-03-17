from ResnetBasics import *

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """

    def __init__(self, in_channels, blocks_sizes,
                 n,block = ResNet753Block, activation='relu', *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=1, activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for k, (in_channels, out_channels) in enumerate(self.in_out_block_sizes)]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_channels=1, blocks_sizes=[128, 64, 32, 16], deepths=[2, 2, 2],
                 activation=nn.ReLU, block=ResNetBasicBlockDecoder, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        self.last_block_index = len(blocks_sizes) - 1
        self.convT = nn.ConvTranspose1d(blocks_sizes[self.last_block_index], in_channels, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.activation_function = nn.ReLU()
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
        # print("s")
        for block in self.blocks:
            x = block(x)
            # print(x.size())
        #print("e")
        #x = self.exit(x)
        x =self.convT(x)
        x = self.batch_norm(x)
        one_before_last = x.clone()
        x = self.activation_function(x)
        # print(x.size())
        # # print(indices.size())
        # x = self.last_layer(x, indices)
        # print("after")
        # print(x.size())
        return x,one_before_last