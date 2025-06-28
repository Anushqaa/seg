import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Reduction
from typing import Dict, Any

class MetricRegistry:
    _metrics = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._metrics[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Unknown metric '{name}'. Available: {available}")
        return cls._metrics[name]

    @classmethod
    def available(cls):
        return sorted(cls._metrics.keys())

@MetricRegistry.register("iou")
class IoU(keras.metrics.Metric):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, smooth: float = 1e-6, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        else:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

@MetricRegistry.register("dice_coefficient")
class DiceCoefficient(keras.metrics.Metric):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, smooth: float = 1e-6, name: str = "dice_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        else:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        self.intersection.assign_add(intersection)
        self.total.assign_add(total)

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.total + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

@MetricRegistry.register("pixel_accuracy")
class PixelAccuracy(keras.metrics.Metric):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, name: str = "pixel_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.correct_pixels = self.add_weight(name="correct_pixels", initializer="zeros")
        self.total_pixels = self.add_weight(name="total_pixels", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        else:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
        correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        self.correct_pixels.assign_add(tf.reduce_sum(correct))
        self.total_pixels.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.correct_pixels / self.total_pixels

    def reset_state(self):
        self.correct_pixels.assign(0.0)
        self.total_pixels.assign(0.0)

@MetricRegistry.register("mean_iou")
class MeanIoU(keras.metrics.Metric):
    def __init__(self, num_classes: int, name: str = "mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name="confusion_matrix", shape=(num_classes, num_classes), initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        if y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        else:
            y_true = tf.cast(y_true, tf.int64)
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        cm = tf.math.confusion_matrix(y_true_f, y_pred_f, num_classes=self.num_classes, dtype=tf.float32)
        self.confusion_matrix.assign_add(cm)

    def result(self):
        diag = tf.linalg.diag_part(self.confusion_matrix)
        row_sum = tf.reduce_sum(self.confusion_matrix, axis=1)
        col_sum = tf.reduce_sum(self.confusion_matrix, axis=0)
        union = row_sum + col_sum - diag
        iou = diag / (union + K.epsilon())
        valid = tf.greater(row_sum, 0)
        iou = tf.where(valid, iou, tf.zeros_like(iou))
        num_valid = tf.reduce_sum(tf.cast(valid, tf.float32))
        return tf.reduce_sum(iou) / (num_valid + K.epsilon())

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

@MetricRegistry.register("precision")
class Precision(keras.metrics.Metric):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, name: str = "precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.true_positives = self.add_weight(name="true_positives", initializer="zeros")
        self.false_positives = self.add_weight(name="false_positives", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        else:
            y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
            y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        return self.true_positives / (self.true_positives + self.false_positives + K.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)

@MetricRegistry.register("recall")
class Recall(keras.metrics.Metric):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, name: str = "recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.true_positives = self.add_weight(name="true_positives", initializer="zeros")
        self.false_negatives = self.add_weight(name="false_negatives", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        else:
            y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
            y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)

@MetricRegistry.register("f1_score")
class F1Score(keras.metrics.Metric):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, name: str = "f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.true_positives = self.add_weight(name="true_positives", initializer="zeros")
        self.false_positives = self.add_weight(name="false_positives", initializer="zeros")
        self.false_negatives = self.add_weight(name="false_negatives", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        else:
            y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
            y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

@MetricRegistry.register("specificity")
class Specificity(keras.metrics.Metric):
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, name: str = "specificity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.true_negatives = self.add_weight(name="true_negatives", initializer="zeros")
        self.false_positives = self.add_weight(name="false_positives", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        else:
            y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
            y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        tn = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + K.epsilon())

    def reset_state(self):
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)

def get_metric(name: str, **kwargs):
    metric_class = MetricRegistry.get(name)
    return metric_class(**kwargs)

def list_metrics():
    return MetricRegistry.available()

def get_metric_info(name: str) -> Dict[str, Any]:
    info = {
        "iou": {"description": "Intersection over Union", "range": "[0,1]"},
        "dice_coefficient": {"description": "Dice coefficient", "range": "[0,1]"},
        "pixel_accuracy": {"description": "Pixel accuracy", "range": "[0,1]"},
        "mean_iou": {"description": "Mean IoU", "range": "[0,1]"},
        "precision": {"description": "Precision", "range": "[0,1]"},
        "recall": {"description": "Recall", "range": "[0,1]"},
        "f1_score": {"description": "F1 score", "range": "[0,1]"},
        "specificity": {"description": "Specificity", "range": "[0,1]"}
    }
    return info.get(name, {"description": "Unknown metric"})
