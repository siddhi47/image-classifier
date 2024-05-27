from image_classifier.architectures import *

models = {
    "resnet": Net_ResNet,
    "dense": Net_DenseNet,
    "inception": Net_Inception,
    "vgg": Net_VGG,
    "mobilenet": Net_MobileNet,
    "vit": Net_ViT
}
