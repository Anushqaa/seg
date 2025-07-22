import os
import matplotlib.pyplot as plt

from seg.utils.data import DataPipeline, get_default_augmentations
from seg.models.base import get_model
from seg.losses import get_loss
from seg.metrics import get_metric

IMAGE_DIR = 'examples/dummy/images'
MASK_DIR  = 'examples/dummy/masks'

input_shape = (1080, 720, 3)

dp = DataPipeline(
    input_shape=input_shape,
    num_classes=1,
    batch_size=2
)

print("\n" + "-"*100 + "\nData Pipeline created!\n" + "-"*100 + "\n")

augs = get_default_augmentations(strong=True)
for fn in augs:
    dp.add_augmentation(fn)

train_ds, val_ds = dp.load_from_directories(
    IMAGE_DIR, MASK_DIR,
    image_ext=".png", mask_ext=".png",
    validation_split=0.2
)
print(train_ds, val_ds)
print("\n" + "-"*100 + "\nDataset created!\n" + "-"*100 + "\n")

model = get_model(
    encoder="resnet50",
    decoder="unet",
    input_shape=input_shape,
    num_classes=1,
    encoder_freeze=True
)

print("\n" + "-"*100 + "\nModel created!\n" + "-"*100 + "\n")

model.compile(
    optimizer="adam",
    loss=get_loss("binary_crossentropy", from_logits=False),
    metrics=[
        get_metric("mean_iou", num_classes=1),
        get_metric("dice_coefficient", num_classes=1)
    ]
)

print("\n" + "-"*100 + "\nModel compiled!\n" + "-"*100 + "\n")
print(model.summary(expand_nested=True))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2
)
try:
    imgs, masks = next(iter(val_ds))
    preds = model.predict(imgs)

    print("\n" + "-"*100 + "\nPredictions made!\n" + "-"*100 + "\n")
finally:
    print("were here")
