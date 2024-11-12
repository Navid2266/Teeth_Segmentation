import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def SegNet_VGG16(input_shape=(512, 512, 3), last_activation='sigmoid'):
    # Input layer
    inputs = Input(shape=input_shape)

    # Load VGG16 model, pretrained on ImageNet, without fully connected layers
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    # Freeze VGG16 layers
    for layer in vgg16.layers:
        layer.trainable = False

    # Encoder using VGG16 layers
    x = vgg16.get_layer("block1_conv2").output  # Output: (256, 256, 64)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)  # Output: (128, 128, 64)

    x = vgg16.get_layer("block2_conv2").output  # Output: (128, 128, 128)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(x)  # Output: (64, 64, 128)

    x = vgg16.get_layer("block3_conv3").output  # Output: (64, 64, 256)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(x)  # Output: (32, 32, 256)

    x = vgg16.get_layer("block4_conv3").output  # Output: (32, 32, 512)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(x)  # Output: (16, 16, 512)

    x = vgg16.get_layer("block5_conv3").output  # Output: (16, 16, 512)
    encoder_output = MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(x)  # Output: (8, 8, 512)

    # Decoder (SegNet style with upsampling)
    
    # Decoder Block 1
    x = UpSampling2D((2, 2))(encoder_output)
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # Decoder Block 2
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # Decoder Block 3
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # Decoder Block 4
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # Decoder Block 5
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # Final convolution to match the input channel size
    outputs = Conv2D(1, (1, 1), activation=last_activation, padding="same")(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model