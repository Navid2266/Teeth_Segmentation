import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def UNET_VGG16(input_shape=(512, 512, 3), last_activation='sigmoid'):
    # Input layer: Accepts a 512x512 RGB image
    inputs = Input(shape=input_shape)

    # Load the VGG16 model, pretrained on ImageNet, with the top (fully connected layers) removed
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)

    # Freeze VGG16 layers to prevent them from being updated during training
    for layer in vgg16.layers:
        layer.trainable = False

    # Encoder (Feature Extraction) using VGG16 layers

    # First block of VGG16: Detects low-level features like edges
    skip1 = vgg16.get_layer("block1_conv2").output  # Shape: (256, 256, 64)

    # Second block of VGG16: Detects slightly more complex features, such as textures
    skip2 = vgg16.get_layer("block2_conv2").output  # Shape: (128, 128, 128)

    # Third block of VGG16: Detects higher-level patterns, such as corners or curves
    skip3 = vgg16.get_layer("block3_conv3").output  # Shape: (64, 64, 256)

    # Fourth block of VGG16: Detects more abstract shapes and patterns
    skip4 = vgg16.get_layer("block4_conv3").output  # Shape: (32, 32, 512)

    # Fifth block of VGG16: Final encoder output; captures the most complex and abstract features
    encoder_output = vgg16.get_layer("block5_conv3").output  # Shape: (16, 16, 512)

    # Decoder (Reconstruction Path)

    # First decoder block: Upsamples the encoded features to a higher resolution
    x = UpSampling2D((2, 2))(encoder_output)  # Upsample from (16, 16) to (32, 32)
    x = concatenate([x, skip4])  # Combine with skip connection from encoder (32, 32, 512 + 512 = 1024 channels)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)  # Convolution to refine features
    x = BatchNormalization()(x)  # Normalizes activations to stabilize training
    x = Dropout(0.3)(x)  # Reduces overfitting by randomly dropping out neurons
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Second decoder block: Further upsampling
    x = UpSampling2D((2, 2))(x)  # Upsample from (32, 32) to (64, 64)
    x = concatenate([x, skip3])  # Combine with skip connection from encoder (64, 64, 256 + 256 = 512 channels)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Third decoder block: Continue upsampling
    x = UpSampling2D((2, 2))(x)  # Upsample from (64, 64) to (128, 128)
    x = concatenate([x, skip2])  # Combine with skip connection from encoder (128, 128, 128 + 128 = 256 channels)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Fourth decoder block: Upsample again
    x = UpSampling2D((2, 2))(x)  # Upsample from (128, 128) to (256, 256)
    x = concatenate([x, skip1])  # Combine with skip connection from encoder (256, 256, 64 + 64 = 128 channels)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Final decoder block: No further upsampling needed, output size should now match input size (512, 512)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # Further reduce channels for final layer
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # Output layer: 1x1 convolution to create a single-channel output with the same spatial size as input
    outputs = Conv2D(1, (1, 1), activation=last_activation, padding='same')(x)  # Shape: (512, 512, 1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
