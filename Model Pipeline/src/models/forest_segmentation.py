"""
mU-Net Model for Forest Segmentation
Extracted from Forest_Segmentation.ipynb

Architecture: Multi-scale U-Net with skip connections from first encoder block
to all decoder levels, enabling better feature reuse.
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.optimizers import Adam


def mUnet_model(img_height=256, img_width=256, img_channels=9):
    """
    mU-Net architecture with multi-scale skip connections.
    
    Args:
        img_height: Input image height (default: 256)
        img_width: Input image width (default: 256)
        img_channels: Number of input channels (default: 9 for Landsat bands)
    
    Returns:
        Compiled Keras Model
    """
    inputs = Input((img_height, img_width, img_channels))
    s = inputs

    # ==========================================
    # Contraction path (Encoder)
    # ==========================================
    
    c1 = Conv2D(75, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(75, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(150, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(150, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(300, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(300, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(600, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(600, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(1200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # ==========================================
    # Expansive path (Decoder) with multi-scale skip connections
    # ==========================================
    
    # Decoder block 1 (16x16 -> 32x32)
    u6 = Conv2DTranspose(600, (2, 2), strides=(2, 2), padding='same')(c5)
    m1 = MaxPooling2D((8, 8))(c1)  # Multi-scale: c1 downsampled 8x
    l1 = concatenate([u6, m1])
    u6 = concatenate([l1, c4])
    c6 = Conv2D(600, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(600, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    # Decoder block 2 (32x32 -> 64x64)
    u7 = Conv2DTranspose(300, (2, 2), strides=(2, 2), padding='same')(c6)
    m2 = MaxPooling2D((4, 4))(c1)  # Multi-scale: c1 downsampled 4x
    l2 = concatenate([u7, m2])
    u7 = concatenate([l2, c3])
    c7 = Conv2D(300, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(300, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    # Decoder block 3 (64x64 -> 128x128)
    u8 = Conv2DTranspose(150, (2, 2), strides=(2, 2), padding='same')(c7)
    m3 = MaxPooling2D((2, 2))(c1)  # Multi-scale: c1 downsampled 2x
    l3 = concatenate([u8, m3])
    u8 = concatenate([l3, c2])
    c8 = Conv2D(150, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(150, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    # Decoder block 4 (128x128 -> 256x256)
    u9 = Conv2DTranspose(75, (2, 2), strides=(2, 2), padding='same')(c8)
    l4 = concatenate([u9, c1])
    u9 = concatenate([l4, c1], axis=3)  # Double skip from c1
    c9 = Conv2D(75, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(75, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary(model):
    """Prints model summary and parameter count."""
    print(f"✅ Model built successfully")
    print(f"📊 Total parameters: {model.count_params():,}")
    return model.count_params()
