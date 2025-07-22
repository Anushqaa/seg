import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np
from seg.models.base import get_model
def test_model(encoder_name="resnet50", decoder_name="unet", 
               input_shape=(128, 128, 3), batch_size=1):
    
    print(f"\n{'='*60}")
    print(f"Testing: {encoder_name} + {decoder_name}")
    print(f"{'='*60}")
    
    model = get_model(
        encoder=encoder_name,
        decoder=decoder_name,
        input_shape=input_shape,
        num_classes=1,
        encoder_freeze=True,
        encoder_weights=None
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    x_dummy = np.random.rand(batch_size, *input_shape).astype(np.float32)
    y_dummy = np.random.randint(0, 2, (batch_size, input_shape[0], input_shape[1], 1)).astype(np.float32)
    
    print(f"\nInput shape: {x_dummy.shape}")
    print(f"Target shape: {y_dummy.shape}")
    
    print("\nTesting forward pass...")
    output = model(x_dummy, training=False)
    print(f"Output shape: {output.shape}")
    
    print("\nTesting single training step...")
    loss, acc = model.train_on_batch(x_dummy, y_dummy)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    print("\nTesting 1 epoch with validation...")
    history = model.fit(
        x_dummy, y_dummy,
        batch_size=batch_size,
        epochs=2,
        validation_data=(x_dummy, y_dummy),
        verbose=1
    )
    
    print("\nModel Summary:")
    model.summary()
    
    del model
    tf.keras.backend.clear_session()
    
    return True

def main():
    
    test_configs = [
        ("resnet50", "unet"),
        ("vgg16", "unetplusplus"),
        ("resnet50", "unetplusplus"),
        ("resnet50", "deeplabv3plus"),
    ]
    
    for encoder, decoder in test_configs:
        try:
            test_model(
                encoder_name=encoder,
                decoder_name=decoder,
                input_shape=(128, 128, 3),
                batch_size=1  
            )
            print(f"✓ {encoder} + {decoder} passed\n")
        except Exception as e:
            print(f"✗ {encoder} + {decoder} failed: {str(e)}\n")

if __name__ == "__main__":
    main()