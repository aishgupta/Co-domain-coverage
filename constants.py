import torchvision.transforms as transforms

PAD_LENGTH = 32
# for text model to unroll layers

PRETRAINED_MODELS = './pretrained_models'

IMAGENET_JPEG_DIR = './NeuraL-Coverage/datasets/ImageNet/' # ILSVRC2012/'
IMAGENET_LABEL_TO_INDEX = './NeuraL-Coverage/datasets/ImageNet/ImageNetLabel2Index.json'
# Since we use the pretrained weights provided by pytorch,
# we should use the same `label_to_index` mapping.

hyper_map = {
    'NLC': None,
    'NC': 0.75,
    'KMNC': 100,
    'SNAC': None,
    'NBC': None,
    'TKNC': 10,
    'TKNP': 50,
    'CC': 10,  	# initially 10 for CIFAR10 and 1000 for ImageNet
    # the maximum likelihood in criterion.access is 2738 and n=1000 (as per original paper)
    'LSC': {'ub': 100, 'n_bins': 1000},
    'DSC': {'ub': 2, 'n_bins': 1000},   	# original paper uses U=2 and n=1000 i.e. bin-length = 2/1000 = 0.002
    'MDSC': 10,
    'CDC': {'n_class': -1, 'n_bins': 100, 'max_per_bin': 10},
    'CDC_v2': {'n_class': -1, 'n_bins': 100, 'max_per_bin': 1}
}

dataset_transforms = {
    "MNIST": {"pixel": [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]		# on Torch Tensor
              ,
              "affine": [  # transforms.RandomErasing(p=1, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.RandomRotation(degrees=15)]
    },

    "FashionMNIST": {"pixel": [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]		# on Torch Tensor
                     ,
                     "affine": [  # transforms.RandomErasing(p=1, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.RandomRotation(degrees=15)]
    },
    "CIFAR10": {"pixel": [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.6)), 						# blur
                          transforms.RandomHorizontalFlip(p=1.0),											# flip
                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                 hue=0.1),  # brightness, noise
                          transforms.RandomAutocontrast(p=1.0)]											# contrast
                # transforms.RandomEqualize(p=1.0)]												# on Torch Tensor
                ,
                "affine": [  # transforms.RandomErasing(p=1, scale=(0.01, 0.03), ratio=(0.3, 3.3), value=0, inplace=False), # erasing
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),  # resized, crop
        transforms.RandomRotation(degrees=15),														# rotation
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0)]									# random perspective
        # transforms.RandomAffine(degrees=0, translate=(0.2,0.2), scale=(0.9, 0.9), shear=10)]		# translation, shear
    },
    "CIFAR100": {"pixel": [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.6)), 									# blur
                           transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                  saturation=0.1, hue=0.1),				# brightness, noise
                           transforms.RandomAutocontrast(p=1.0)]														# contrast
                 # transforms.RandomEqualize(p=1.0)]		# on Torch Tensor
                 ,
                 "affine": [  # transforms.RandomErasing(p=1, scale=(0.01, 0.03), ratio=(0.3, 3.3), value=0, inplace=False), # erasing
        transforms.RandomHorizontalFlip(p=1.0),														# flip
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),  # resized, crop
        transforms.RandomRotation(degrees=15),														# rotation
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0)]									# random perspective
        # transforms.RandomAffine(degrees=0, translate=(0.2,0.2), scale=(0.9, 0.9), shear=10)]			# translation, shear
    },
    "SVHN":     {"pixel": [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.6)), 						# blur
                           transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                  saturation=0.2, hue=0.2),  # brightness, noise
                           ]												# on Torch Tensor
                 ,
                 "affine": [transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),  # resized, crop
                            transforms.RandomRotation(degrees=15),														# rotation
                            transforms.RandomPerspective(distortion_scale=0.1, p=1.0)]									# random perspective
                 },
    "ImageNet": {"pixel": [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.6)), 						# blur
                          transforms.RandomHorizontalFlip(p=1.0),											# flip
                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                 hue=0.1),  # brightness, noise
                          transforms.RandomAutocontrast(p=1.0)]											# contrast
                # transforms.RandomEqualize(p=1.0)]												# on Torch Tensor
                ,
                "affine": [  # transforms.RandomErasing(p=1, scale=(0.01, 0.03), ratio=(0.3, 3.3), value=0, inplace=False), # erasing
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),  # resized, crop
        transforms.RandomRotation(degrees=15),														# rotation
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0)]									# random perspective
        # transforms.RandomAffine(degrees=0, translate=(0.2,0.2), scale=(0.9, 0.9), shear=10)]		# translation, shear
    },

}

data_info = {"MNIST": {'img_size': 28, 'n_img_channels': 1, 'n_class': 10, 'n_test_class': 10},
             "FashionMNIST": {'img_size': 28, 'n_img_channels': 1, 'n_class': 10, 'n_test_class': 10},
             "SVHN": {'img_size': 32, 'n_img_channels': 3, 'n_class': 10, 'n_test_class': 10},
             "CIFAR10": {'img_size': 32, 'n_img_channels': 3, 'n_class': 10, 'n_test_class': 10},
             "CIFAR100": {'img_size': 32, 'n_img_channels': 3, 'n_class': 100, 'n_test_class': 100},
             "ImageNet": {'img_size': 128, 'n_img_channels': 3, 'n_class': 1000, 'n_test_class': 100},
             }
