import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from seg.models.base import get_model

def run_tiny_training_and_prediction(
    encoder_name="resnet50", decoder_name="unet",
    input_shape=(64, 64, 3), batch_size=1,
    num_classes=1, epochs=2
):
    print(f"\nTesting: {encoder_name} + {decoder_name}\n{'='*40}")
    model = get_model(
        encoder=encoder_name,
        decoder=decoder_name,
        input_shape=input_shape,
        num_classes=num_classes,
        encoder_freeze=True,
        encoder_weights=None,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    x_train = np.random.rand(batch_size * 2, *input_shape).astype(np.float32)
    y_train = np.random.randint(0, 2, (batch_size * 2, input_shape[0], input_shape[1], 1)).astype(np.float32)

    x_val = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_val = np.random.randint(0, 2, (batch_size, input_shape[0], input_shape[1], 1)).astype(np.float32)

    print("Training model...")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=2
    )

    print("Predicting on validation image...")
    preds = model.predict(x_val)
    print(f"Prediction shape: {preds.shape}")

    model(x_val)
    print("\nModel Summary:")
    model.summary()

    return preds

if __name__ == "__main__":
    run_tiny_training_and_prediction()

