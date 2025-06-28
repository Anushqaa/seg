import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from typing import Tuple, List, Optional, Union, Dict, Any, Callable
import functools
import warnings

class DataPipeline:
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 3),
        num_classes: int = 1,
        batch_size: int = 16,
        shuffle_buffer_size: int = 1000,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        num_parallel_calls: int = tf.data.AUTOTUNE
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.augmentation_functions = []

    def add_augmentation(self, aug_func: Callable) -> None:
        self.augmentation_functions.append(aug_func)

    def load_from_directories(
        self,
        image_dir: str,
        mask_dir: str,
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        validation_split: float = 0.2,
        seed: int = 42
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        image_paths = self._get_file_paths(image_dir, image_ext)
        mask_paths = self._get_file_paths(mask_dir, mask_ext)
        image_paths, mask_paths = self._match_files(image_paths, mask_paths)
        train_images, val_images, train_masks, val_masks = self._train_val_split(
            image_paths, mask_paths, validation_split, seed
        )
        train_dataset = self._create_dataset(train_images, train_masks, training=True)
        val_dataset = self._create_dataset(val_images, val_masks, training=False)
        return train_dataset, val_dataset

    def _get_file_paths(self, directory: str, extension: str) -> List[str]:
        files = [
            os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(extension.lower())
        ]
        return files

    def _match_files(
        self,
        image_paths: List[str],
        mask_paths: List[str]
    ) -> Tuple[List[str], List[str]]:
        image_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
        mask_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
        common = sorted(set(image_basenames) & set(mask_basenames))
        if not common:
            raise ValueError("No matching image-mask pairs found")
        imgs = [image_basenames[n] for n in common]
        masks = [mask_basenames[n] for n in common]
        return imgs, masks

    def _train_val_split(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        validation_split: float,
        seed: int
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        np.random.seed(seed)
        idx = np.random.permutation(len(image_paths))
        split = int(len(idx) * (1 - validation_split))
        train_idx, val_idx = idx[:split], idx[split:]
        return (
            [image_paths[i] for i in train_idx],
            [image_paths[i] for i in val_idx],
            [mask_paths[i] for i in train_idx],
            [mask_paths[i] for i in val_idx]
        )

    def _create_dataset(self, image_paths, mask_paths, training) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        ds = ds.map(self._load_and_decode, num_parallel_calls=self.num_parallel_calls)
        ds = self._apply_pipeline(ds, training)
        return ds

    def _load_and_decode(self, image_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.read_file(image_path)
        msk = tf.io.read_file(mask_path)
        img = tf.io.decode_image(img, channels=self.input_shape[2], expand_animations=False)
        msk = tf.io.decode_image(msk, channels=1, expand_animations=False)
        img.set_shape([None, None, self.input_shape[2]])
        msk.set_shape([None, None, 1])
        img = tf.image.resize(img, self.input_shape[:2], method='bilinear')
        msk = tf.image.resize(msk, self.input_shape[:2], method='nearest')
        img = tf.cast(img, tf.float32) / 255.0
        if self.num_classes > 1:
            msk = tf.cast(msk, tf.int32)
            msk = tf.one_hot(tf.squeeze(msk, -1), self.num_classes)
        else:
            msk = tf.cast(msk, tf.float32) / 255.0
        return img, msk

    def _apply_pipeline(self, dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
        if training and self.augmentation_functions:
            for fn in self.augmentation_functions:
                dataset = dataset.map(fn, num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.cache()
        if training:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        return dataset

class Augmentation:
    @staticmethod
    def random_flip_horizontal(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        return image, mask

    @staticmethod
    def random_flip_vertical(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        return image, mask

    @staticmethod
    def random_brightness(image: tf.Tensor, mask: tf.Tensor, max_delta: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_brightness(image, max_delta)
        return image, mask

    @staticmethod
    def random_contrast(image: tf.Tensor, mask: tf.Tensor, lower: float = 0.8, upper: float = 1.2) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_contrast(image, lower, upper)
        return image, mask

    @staticmethod
    def random_saturation(image: tf.Tensor, mask: tf.Tensor, lower: float = 0.8, upper: float = 1.2) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.random_saturation(image, lower, upper)
        return image, mask

class Visualization:
    @staticmethod
    def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, colormap: str = "jet") -> np.ndarray:
        import matplotlib.pyplot as plt
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        if image.max() > 1:
            image = image / 255.0
        cmap = plt.get_cmap(colormap)
        colored = cmap(mask)[:, :, :3]
        overlay = (1 - alpha) * image + alpha * colored
        return np.clip(overlay, 0, 1)

    @staticmethod
    def plot_samples(dataset: tf.data.Dataset, num_samples: int = 4, figsize: Tuple[int, int] = (15, 10)) -> None:
        import matplotlib.pyplot as plt
        ds = dataset.unbatch().take(num_samples)
        fig, axes = plt.subplots(2, num_samples, figsize=figsize)
        for i, (img, msk) in enumerate(ds):
            img_np = img.numpy()
            msk_np = msk.numpy()
            axes[0, i].imshow(img_np)
            axes[0, i].axis("off")
            axes[1, i].imshow(msk_np.squeeze(), cmap="gray")
            axes[1, i].axis("off")
        plt.tight_layout()
        plt.show()

def get_default_augmentations(strong: bool = False) -> List[Callable]:
    augmentations = [
        Augmentation.random_flip_horizontal,
        Augmentation.random_flip_vertical,
    ]
    if strong:
        augmentations.extend([
            Augmentation.random_brightness,
            Augmentation.random_contrast,
            Augmentation.random_saturation,
        ])
    return augmentations