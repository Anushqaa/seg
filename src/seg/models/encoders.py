# src/seg/models/encoders.py - COMPLETE CORRECTED VERSION

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers, applications
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings

class EncoderRegistry:
    _encoders = {}

    @classmethod
    def register(cls, name: str, feature_layers: list):
        def decorator(f):
            cls._encoders[name] = (f, feature_layers)
            return f
        return decorator
    
    @classmethod
    def get(cls, name: str):
        if name not in cls._encoders:
            available = list(cls._encoders.keys())
            raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
        return cls._encoders[name][0]  # Return builder function
    
    @classmethod
    def get_feature_layers(cls, name: str) -> List[str]:
        if name not in cls._encoders:
            raise ValueError(f"Encoder '{name}' not registered")
        return cls._encoders[name][1]  # Return feature layers
    
    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._encoders.keys())

def _create_backbone(
    backbone: Model,
    feature_layers: List[str],
    input_shape: Tuple[int, int, int]
) -> Model:
    """### FIXED: Returns a Model that outputs List[Tensor] consistently"""
    # Auto-detect layers if not specified
    if not feature_layers:
        # Find all convolutional outputs
        feature_layers = [layer.name for layer in backbone.layers 
                          if isinstance(layer, layers.Conv2D) and 'pool' not in layer.name]
        # Take the last 5 conv layers or whatever available
        feature_layers = feature_layers[-5:] if len(feature_layers) >= 5 else feature_layers
        warnings.warn(f"Using auto-detected feature layers: {feature_layers}")
    
    feature_outputs = []
    for layer_name in feature_layers:
        try:
            layer = backbone.get_layer(layer_name)
            feature_outputs.append(layer.output)
        except ValueError:
            warnings.warn(f"Layer '{layer_name}' not found in backbone")
    
    # Fallback to final layer if no features found
    if not feature_outputs:
        feature_outputs = [backbone.output]
        warnings.warn("No valid feature layers found. Using final output layer")
    
    # ### FIXED: Create wrapper that ensures call() always returns List[Tensor]
    feature_model = Model(inputs=backbone.input, outputs=feature_outputs, name=f"{backbone.name}_features")
    
    class FeatureExtractor(Model):
        def __init__(self, base_model: Model):
            super().__init__(name=base_model.name + "_wrapper")
            self.base_model = base_model
            
        def call(self, inputs, training=None):
            outputs = self.base_model(inputs, training=training)
            # Ensure output is always a list
            if not isinstance(outputs, list):
                outputs = [outputs]
            return outputs
    
    return FeatureExtractor(feature_model)

# ============== Encoder Implementations ============== #

@EncoderRegistry.register("resnet50", [
    "conv1_relu", "conv2_block3_out", "conv3_block4_out", 
    "conv4_block6_out", "conv5_block3_out"
])
def build_resnet50(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("resnet50"), 
        input_shape
    )

@EncoderRegistry.register("resnet101", [
    "conv1_relu", "conv2_block3_out", "conv3_block4_out",
    "conv4_block23_out", "conv5_block3_out"
])
def build_resnet101(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.ResNet101(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("resnet101"), 
        input_shape
    )

@EncoderRegistry.register("vgg16", [
    "block1_conv2", "block2_conv2", "block3_conv3",
    "block4_conv3", "block5_conv3"
])
def build_vgg16(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.VGG16(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("vgg16"), 
        input_shape
    )

@EncoderRegistry.register("vgg19", [
    "block1_conv2", "block2_conv2", "block3_conv4", 
    "block4_conv4", "block5_conv4"
])
def build_vgg19(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.VGG19(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("vgg19"), 
        input_shape
    )

@EncoderRegistry.register("efficientnet-b0", [
    "stem_activation", "block2a_expand_activation", 
    "block3a_expand_activation", "block4a_expand_activation", 
    "block6a_expand_activation"
])
def build_efficientnet_b0(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("efficientnet-b0"), 
        input_shape
    )

@EncoderRegistry.register("efficientnet-b1", [
    "stem_activation", "block2a_expand_activation", 
    "block3a_expand_activation", "block4a_expand_activation", 
    "block6a_expand_activation"
])
def build_efficientnet_b1(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.EfficientNetB1(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("efficientnet-b1"), 
        input_shape
    )

@EncoderRegistry.register("mobilenet_v2", [
    "block_1_expand_relu", "block_3_expand_relu", 
    "block_6_expand_relu", "block_13_expand_relu", 
    "out_relu"
])
def build_mobilenet_v2(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("mobilenet_v2"), 
        input_shape
    )

@EncoderRegistry.register("densenet121", [
    "conv1/relu", "pool2_pool", "pool3_pool", 
    "pool4_pool", "relu"
])
def build_densenet121(
    input_shape: Tuple[int, int, int],
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    backbone = applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        **kwargs
    )
    return _create_backbone(
        backbone, 
        EncoderRegistry.get_feature_layers("densenet121"), 
        input_shape
    )

@EncoderRegistry.register("custom", [])
def build_custom_encoder(
    input_shape: Tuple[int, int, int],
    filters: List[int] = [64, 128, 256, 512, 1024],
    **kwargs
) -> Model:
    return CustomEncoder(
        input_shape=input_shape,
        filters=filters,
        **kwargs
    )

# ============== Custom Encoder Class ============== #

class CustomEncoder(Model):
    """### FIXED: Returns List[Tensor] consistently like other encoders"""
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        filters: List[int] = [64, 128, 256, 512, 1024],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_shape_val = input_shape
        self.filters = filters

        self.blocks = []
        for i, filter_count in enumerate(filters):
            block = self._build_encoder_block(
                filter_count, 
                name=f"encoder_block_{i+1}"
            )
            self.blocks.append(block)

    def _build_encoder_block(self, filters: int, name: str) -> Sequential:
        return Sequential([
            layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1"),
            layers.BatchNormalization(name=f"{name}_bn1"),
            layers.ReLU(name=f"{name}_relu1"),
            layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2"),
            layers.BatchNormalization(name=f"{name}_bn2"), 
            layers.ReLU(name=f"{name}_relu2"),
            layers.MaxPooling2D(2, name=f"{name}_pool")
        ], name=name)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> List[tf.Tensor]:
        features = []
        x = inputs
        
        for block in self.blocks:
            x = block(x, training=training)
            features.append(x)
            
        return features

# ============== Public API Functions ============== #

def get_encoder(
    name: str,
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    weights: Optional[str] = "imagenet",
    **kwargs
) -> Model:
    encoder_builder = EncoderRegistry.get(name)
    
    if name == "custom":
        return encoder_builder(input_shape=input_shape, **kwargs)
    
    return encoder_builder(
        input_shape=input_shape,
        weights=weights,
        **kwargs
    )

def list_encoders() -> List[str]:
    return EncoderRegistry.available()

def get_encoder_info(name: str) -> Dict[str, Any]:
    encoder_info = {
        "resnet50": {
            "description": "ResNet50 with ImageNet pre-training",
            "parameters": "25.6 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "resnet101": {
            "description": "ResNet101 with ImageNet pre-training",
            "parameters": "44.7 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "vgg16": {
            "description": "VGG16 with ImageNet pre-training",
            "parameters": "14.7 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "vgg19": {
            "description": "VGG19 with ImageNet pre-training",
            "parameters": "20.0 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "efficientnet-b0": {
            "description": "EfficientNet-B0 with ImageNet pre-training",
            "parameters": "5.3 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "efficientnet-b1": {
            "description": "EfficientNet-B1 with ImageNet pre-training",
            "parameters": "7.8 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "mobilenet_v2": {
            "description": "MobileNetV2 with ImageNet pre-training",
            "parameters": "3.5 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "densenet121": {
            "description": "DenseNet121 with ImageNet pre-training",
            "parameters": "8.1 million",
            "input_shape": ">= 32x32",
            "feature_levels": 5
        },
        "custom": {
            "description": "Custom encoder with configurable filters",
            "parameters": "Depends on filter configuration",
            "input_shape": "Any",
            "feature_levels": "Variable (based on filter count)"
        }
    }
    
    if name not in encoder_info:
        available = list(encoder_info.keys())
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
    
    # Add feature layers to info
    info = encoder_info[name]
    try:
        info["feature_layers"] = EncoderRegistry.get_feature_layers(name)
    except ValueError:
        info["feature_layers"] = "Auto-detected"
    
    return info