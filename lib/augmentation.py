import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ])


def get_validation_augmentation():
    return A.Compose([
        ToTensorV2(),
    ])
