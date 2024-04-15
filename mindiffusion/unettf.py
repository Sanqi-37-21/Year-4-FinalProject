from tensorflow.keras import Model
from functools import partial
from tqdm import tqdm
import tensorflow as tf

# https://arxiv.org/pdf/1803.08494v3.pdf  change the [N,C,H,W] to [N,H,W,C]
class GroupNorm(tf.keras.layers.Layer):
    def __init__(self, groups: int, channels: int, eps=1e-5, name=None) -> None:
        super().__init__()
        self.groups = groups
        self.eps = eps
        # one for gamma found in PyTorch documentation called weights
        self.gamma = self.add_weight(name='gamma', shape=(1, 1, 1, channels), initializer='ones', trainable=True)
        # zero for bet found in PyTorch documentation called bias
        self.beta = self.add_weight(name='beta', shape=(1, 1, 1, channels), initializer='zeros', trainable=True)
    def call(self, x):
        # x: input features with shape [N,H,W,C]
        # gamma, beta: scale and offset, with shape [1,1,1,C]
        N, H, W, C = x.shape
        # reshape to number of groups with channel/group channels
        x = tf.reshape(x, [N, H, W, self.groups, C // self.groups])
        # calculate varience and mean for H, W, C // groups
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        # normalize the original data
        x = (x - mean) / tf.sqrt(var + self.eps)
        # reshape back to normal form of tensor
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta

class Conv3(tf.keras.Model):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        # sequential need add []
        self.main = tf.keras.Sequential([
            # Conv2d
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
            GroupNorm(groups=8, channels=out_channels),
            # relu
            tf.keras.layers.ReLU(),
        ])
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
            GroupNorm(groups=8, channels=out_channels),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
            GroupNorm(groups=8, channels=out_channels),
            tf.keras.layers.ReLU(),
        ])

        self.is_res = is_res

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(tf.keras.Model):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), tf.keras.layers.MaxPooling2D(2)]
        self.model = tf.keras.Sequential([*layers])

    def call(self, x: tf.Tensor) -> tf.Tensor:

        return self.model(x)


class UnetUp(tf.keras.Model):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            # nn.ConvTranspose2d
            tf.keras.layers.Conv2DTranspose(filters=out_channels, kernel_size=2, strides=2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = tf.keras.Sequential([*layers])

    def call(self, x: tf.Tensor, skip: tf.Tensor) -> tf.Tensor:
        # tensorflow 中 concat 的 channel 在 -1
        # torch.cat
        x = tf.concat((x, skip), -1)
        x = self.model(x)

        return x


class TimeSiren(tf.keras.Model):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()
        # nn.Linear
        self.lin1 = tf.keras.layers.Dense(emb_dim, use_bias=False, input_shape=(1,))
        self.lin2 = tf.keras.layers.Dense(emb_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # 没有 view
        x = tf.reshape(x, [-1, 1])
        # torch.sin
        x = tf.math.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(tf.keras.Model):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = tf.keras.Sequential([tf.keras.layers.AveragePooling2D(pool_size=(4, 4)), tf.keras.layers.ReLU()])

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=2 * n_feat, kernel_size=4, strides=4),
            GroupNorm(groups=8, channels=2 * n_feat),
            tf.keras.layers.ReLU(),
        ])

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = tf.keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, strides=(1, 1), padding='same')

    def call(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:

        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        thro = self.to_vec(down3)
        temb = tf.reshape(self.timeembed(t), [-1, 1, 1, self.n_feat * 2])

        thro = self.up0(thro + temb)

        up1 = self.up1(thro, down3) + temb
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.out(tf.concat((up3, x), -1))

        return out