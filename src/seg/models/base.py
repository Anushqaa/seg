# src/seg/models/base.py - COMPLETE CORRECTED VERSION

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings

from .encoders import get_encoder
from .decoders import get_decoder

class BaseModel(Model):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        decoder_name: str = "unet",
        num_classes: int = 1,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        encoder_weights: Optional[str] = "imagenet",
        encoder_freeze: bool = False,
        decoder_config: Optional[Dict[str, Any]] = None,
        activation: str = "sigmoid",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._validate_input(encoder_name, decoder_name, num_classes, input_shape, activation)
        
        # Initialize parameters
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.encoder_weights = encoder_weights
        self.encoder_freeze = encoder_freeze
        self.decoder_config = decoder_config or {}
        self.activation_name = activation
        self._is_compiled = False
        self._decoder_built = False  # ### FIXED: Track decoder building state

        # Build model components
        self._build_encoder()
        # ### FIXED: Don't build decoder yet - wait for first call
        self._build_output_layer()

    def _validate_input(
            self, 
            encoder_name: str,
            decoder_name: str, 
            num_classes: int, 
            input_shape: Tuple[int, int, int], 
            activation: str
        ) -> None:
        # Validate activation/class mismatch
        if num_classes == 1 and activation == "softmax":
            raise ValueError("Use 'sigmoid' not 'softmax' for binary segmentation")
        elif num_classes > 1 and activation == "sigmoid":
            raise ValueError("Use 'softmax' not 'sigmoid' for multi-class segmentation")
            
        # Validate decoder compatibility
        if decoder_name == "deeplabv3plus" and not encoder_name.startswith(("resnet", "vgg", "densenet")):
            warnings.warn(f"DeepLabV3+ may not work optimally with {encoder_name} encoder")
            
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must have 3 dimensions (H, W, C), got {input_shape}")
            
        if input_shape[0] < 32 or input_shape[1] < 32:
            warnings.warn("Input dimensions < 32 may cause issues with some encoders")

    def _build_encoder(self) -> None:
        self.encoder = get_encoder(
            name=self.encoder_name,
            input_shape=self.input_shape,
            weights=self.encoder_weights
        )
        
        if self.encoder_freeze:
            self.encoder.trainable = False

    def _build_decoder(self, encoder_outputs: List[tf.Tensor]) -> None:
        """### FIXED: Build decoder with actual encoder outputs, not dummy input"""
        if not isinstance(encoder_outputs, list):
            encoder_outputs = [encoder_outputs]
            
        self.decoder = get_decoder(
            name=self.decoder_name,
            num_encoder_levels=len(encoder_outputs),
            num_classes=self.num_classes,
            **self.decoder_config
        )
        self._decoder_built = True

    def _build_output_layer(self) -> None:
        if self.activation_name == "sigmoid":
            self.final_activation = layers.Activation("sigmoid", name="sigmoid_activation")
        elif self.activation_name == "softmax":
            self.final_activation = layers.Activation("softmax", name="softmax_activation")
        elif self.activation_name == "linear" or self.activation_name is None:
            self.final_activation = layers.Activation("linear", name="linear_activation")
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Get encoder features (list of tensors)
        encoder_outputs = self.encoder(inputs, training=training)
        
        # ### FIXED: Ensure output is always list format
        if not isinstance(encoder_outputs, list):
            encoder_outputs = [encoder_outputs]
        
        # ### FIXED: Build decoder lazily on first call with real feature maps
        if not self._decoder_built:
            self._build_decoder(encoder_outputs)
            
        # Process through decoder and activation
        decoder_output = self.decoder(encoder_outputs, target_size=tf.shape(inputs)[1:3], training=training)
        return self.final_activation(decoder_output)
    
    def compile(
            self,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
            loss: Union[str, tf.keras.losses.Loss] = "binary_crossentropy",
            metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None,
            **kwargs
    ) -> None:
        if metrics is None:
            metrics = ["accuracy"]
        
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        self._is_compiled = True

    def summary(self, **kwargs) -> None:
        # ### FIXED: Build model completely before summary
        if not self.built or not self._decoder_built:
            dummy_input = tf.keras.Input(shape=self.input_shape)
            _ = self(dummy_input)  # This triggers full building
        
        print(f"\n{'='*50}")
        print(f"Segmentation Model Summary")
        print(f"{'='*50}")
        print(f"Encoder: {self.encoder_name}")
        print(f"Decoder: {self.decoder_name}")
        print(f"Input Shape: {self.input_shape}")
        print(f"Output Classes: {self.num_classes}")
        print(f"Final Activation: {self.activation_name}")
        print(f"Encoder Frozen: {self.encoder_freeze}")
        print(f"{'='*50}\n")
        
        super().summary(**kwargs)

def get_model(
    encoder: str = "resnet50",
    decoder: str = "unet", 
    num_classes: int = 1,
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    **kwargs
) -> BaseModel:  
    return BaseModel(
        encoder_name=encoder,
        decoder_name=decoder,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )