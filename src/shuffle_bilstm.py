import tensorflow as tf
from tensorflow.keras import layers, models

class GroupConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides, groups, **kwargs):
        super(GroupConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.groups = groups
        self.convs = []
        
    def build(self, input_shape):
        for _ in range(self.groups):
            self.convs.append(
                layers.Conv2D(
                    self.filters // self.groups,
                    self.kernel_size,
                    strides=self.strides,
                    padding='same'
                )
            )
        super(GroupConv2D, self).build(input_shape)
    
    def call(self, x):
        split_inputs = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        group_outputs = [conv(split) for conv, split in zip(self.convs, split_inputs)]
        return tf.concat(group_outputs, axis=-1)

class ChannelShuffle(layers.Layer):
    def __init__(self, groups, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = x.shape[-1]
        channels_per_group = channels // self.groups
        
        x = tf.reshape(x, [batch_size, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        return x

def shuffle_unit(x, filters=112, groups=4):
    shortcut = x
    
    # Main path
    x = GroupConv2D(4*28, (1,1), (1,1), groups)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = ChannelShuffle(groups)(x)
    x = GroupConv2D(filters, (3,3), (2,2), groups)(x)
    x = layers.BatchNormalization()(x)
    x = GroupConv2D(4*28, (1,1), (1,1), groups)(x)
    x = layers.BatchNormalization()(x)
    
    # Projection shortcut
    shortcut = layers.Conv2D(4*28, (1,1), strides=(2,2), padding='same')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)
    
    # Add
    x = layers.Add()([x, shortcut])
    return x

def build_shuffle_bilstm(input_shape=(256,256,3), n_classes=3):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(24, (3,3), strides=(2,2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    for _ in range(3):
        x = shuffle_unit(x)
    
    x = layers.Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs, x)
    return model
