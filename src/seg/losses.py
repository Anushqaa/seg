import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Reduction
from typing import Dict, Any, Union
import warnings

class LossRegistry:
    _losses = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._losses[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._losses:
            available = list(cls._losses.keys())
            raise ValueError(f"Unknown loss '{name}'. Available: {available}")
        return cls._losses[name]

    @classmethod
    def available(cls):
        return sorted(cls._losses.keys())


def _smooth_labels(y_true: tf.Tensor, smoothing: float = 0.1) -> tf.Tensor:
    if smoothing <= 0:
        return y_true
    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
    smooth_positives = 1.0 - smoothing
    smooth_negatives = smoothing / num_classes
    return y_true * smooth_positives + smooth_negatives

@LossRegistry.register("binary_crossentropy")
class BinaryCrossentropy(keras.losses.Loss):
    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        reduction: Reduction = Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "binary_crossentropy",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if self.label_smoothing > 0:
            y_true = _smooth_labels(y_true, self.label_smoothing)
        return tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )

@LossRegistry.register("categorical_crossentropy")
class CategoricalCrossentropy(keras.losses.Loss):
    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        reduction: Reduction = Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "categorical_crossentropy",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        if self.label_smoothing > 0:
            y_true = _smooth_labels(y_true, self.label_smoothing)
        return tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )

@LossRegistry.register("dice_loss")
class DiceLoss(keras.losses.Loss):
    def __init__(
        self,
        smooth: float = 1e-6,
        squared_pred: bool = False,
        reduction: Reduction = Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "dice_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.smooth = smooth
        self.squared_pred = squared_pred

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        if self.squared_pred:
            y_pred_f = tf.square(y_pred_f)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        dice_coef = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice_coef

@LossRegistry.register("iou_loss")
class IoULoss(keras.losses.Loss):
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: Reduction = Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "iou_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.smooth = smooth

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        iou_coef = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou_coef

@LossRegistry.register("focal_loss")
class FocalLoss(keras.losses.Loss):
    def __init__(
        self,
        alpha: Union[float, tf.Tensor] = 0.25,
        gamma: float = 2.0,
        from_logits: bool = False,
        reduction: Reduction = Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "focal_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ce_loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        focal_loss = focal_weight * ce_loss
        return focal_loss

@LossRegistry.register("tversky_loss")
class TverskyLoss(keras.losses.Loss):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6,
        reduction: Reduction = Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "tversky_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        TP = tf.reduce_sum(y_true_f * y_pred_f)
        FN = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        FP = tf.reduce_sum((1 - y_true_f) * y_pred_f)
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1.0 - tversky_index

@LossRegistry.register("combined_loss")
class CombinedLoss(keras.losses.Loss):
    def __init__(
        self,
        losses: Dict[str, float],
        reduction: Reduction = Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "combo_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.losses = losses
        self.loss_functions = {}
        for loss_name, weight in losses.items():
            if loss_name in LossRegistry._losses:
                self.loss_functions[loss_name] = LossRegistry.get(loss_name)()
            else:
                warnings.warn(f"Unknown loss function: {loss_name}")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        total_loss = 0.0
        for loss_name, weight in self.losses.items():
            if loss_name in self.loss_functions:
                loss_value = self.loss_functions[loss_name](y_true, y_pred)
                total_loss += weight * loss_value
        return total_loss

def get_loss(
    name: str,
    **kwargs
) -> keras.losses.Loss:
    if name == "combined_loss":
        return CombinedLoss(**kwargs)
    loss_class = LossRegistry.get(name)
    return loss_class(**kwargs)

def list_losses():
    return LossRegistry.available()

def get_loss_info(name: str) -> Dict[str, Any]:
    if name not in LossRegistry._losses:
        raise ValueError(f"Unknown loss '{name}'. Available: {LossRegistry.available()}")
    loss_info = {
        "binary_crossentropy": {
            "description": "Standard binary cross-entropy for binary segmentation",
            "use_cases": ["Binary segmentation", "General purpose"],
            "strengths": ["Simple", "Well-studied", "Fast"]
        },
        "categorical_crossentropy": {
            "description": "Categorical cross-entropy for multi-class segmentation",
            "use_cases": ["Multi-class segmentation"],
            "strengths": ["Standard for classification", "Probabilistic interpretation"]
        },
        "dice_loss": {
            "description": "Dice coefficient loss for overlap optimization",
            "use_cases": ["Medical imaging", "Imbalanced classes"],
            "strengths": ["Handles class imbalance", "Direct overlap optimization"]
        },
        "iou_loss": {
            "description": "IoU (Jaccard) loss for intersection over union optimization",
            "use_cases": ["Object detection", "Segmentation"],
            "strengths": ["Direct IoU optimization", "Scale invariant"]
        },
        "focal_loss": {
            "description": "Focal loss for hard negative mining",
            "use_cases": ["Class imbalance", "Hard example mining"],
            "strengths": ["Addresses class imbalance", "Focuses on hard examples"]
        },
        "tversky_loss": {
            "description": "Generalized Dice loss with controllable precision/recall",
            "use_cases": ["Medical imaging", "Precision/recall trade-offs"],
            "strengths": ["Controllable precision/recall", "Generalizes Dice"]
        },
        "combined_loss": {
            "description": "Combination of multiple loss functions",
            "use_cases": ["Multi-objective optimization"],
            "strengths": ["Flexible", "Combines advantages of multiple losses"]
        }
    }
    return loss_info.get(name, {"description": "Unknown loss function"})
