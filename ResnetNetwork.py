from ResnetBasics import *

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels = 1, blocks_sizes=[16, 32, 64, 128], deepths=[2, 2, 2, 2],
                 activation=nn.ReLU,block=ResNetBasicBlockEncoder, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, self.blocks_sizes[0], kernel_size= 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

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

        self.exit = nn.Sequential(
            nn.ConvTranspose1d(blocks_sizes[self.last_block_index], in_channels, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False),
            nn.BatchNorm1d(in_channels),
            activation()
        )
        self.last_layer = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)

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
        for block in self.blocks:
            x = block(x)
        x = self.exit(x)
        one_before_last = x.copy()
        x = self.last_layer(x)
        return x,one_before_last