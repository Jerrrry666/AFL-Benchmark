import torchvision


def resnet152_domainnet(args):
    return torchvision.models.resnet152(weights=None, num_classes=args.class_num)