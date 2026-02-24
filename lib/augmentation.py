import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ])


def get_validation_augmentation():
    return A.Compose([
        ToTensorV2(),
    ])
