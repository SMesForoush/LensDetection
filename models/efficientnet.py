"""EfficientNet-B0 (Tan & Le 2019) for lens classification, built with timm and trained from random initialisation.

    model = build_efficientnet(in_channels=4)   # 4,010,398 parameters, 2 output logits

timm adapts the stem to any number of input channels, and the network is fully convolutional up to the global average
pool, so the same weights work at any input size (the thesis uses 300x300).
"""
import timm


def build_efficientnet(in_channels=4, num_classes=2, dropout=0.2, drop_path_rate=0.2, variant="efficientnet_b0"):
    """dropout: before the classifier; drop_path_rate: stochastic depth. Both default to the B0 values of the paper."""
    return timm.create_model(variant, pretrained=False, in_chans=in_channels, num_classes=num_classes,
                             drop_rate=dropout, drop_path_rate=drop_path_rate)
