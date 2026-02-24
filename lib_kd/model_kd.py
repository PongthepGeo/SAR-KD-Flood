import segmentation_models_pytorch as smp
from lib.model import PSPMixer


def create_teacher_model(model_type="PSPNet", encoder="mobilenet_v2",
                         encoder_weights="imagenet", in_channels=1, classes=1):
    if model_type == "Unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                        in_channels=in_channels, classes=classes, activation=None)
    elif model_type == "PSPNet":
        return smp.PSPNet(encoder_name=encoder, encoder_weights=encoder_weights,
                          in_channels=in_channels, classes=classes, activation=None)
    else:
        raise ValueError(f"Unknown teacher type: {model_type}")


def create_student_model(in_channels=1, num_classes=1, patch=32,
                         hidden=128, depth=3, tokens_mlp=64, channels_mlp=256):
    return PSPMixer(in_ch=in_channels, num_classes=num_classes,
                    patch=patch, hidden=hidden, depth=depth,
                    tokens_mlp=tokens_mlp, channels_mlp=channels_mlp, img_size=256)
