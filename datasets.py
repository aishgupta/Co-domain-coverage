from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST
from torchvision import transforms
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import pickle
import copy
import os


image_transforms = {
    "MNIST": transforms.Compose([transforms.ToTensor()]),
    "FashionMNIST": transforms.Compose([transforms.ToTensor()]),
    "CIFAR10": transforms.Compose([transforms.ToTensor()]),
    "CIFAR100": transforms.Compose([transforms.ToTensor()]),
    "SVHN": transforms.Compose([transforms.ToTensor()])
}


class Data:
    """
    =====================================================================================
    Customised dataloader class for vision dataset

    samples_per_class: for reduced training set with #images_per_class = samples_per_class
    class_indices    : list of classes to be considered. If it is ['all'], then all the
                                       classes will be considered
    ======================================================================================
    """

    def __init__(self, dataset_name, root, train, normalise_imgs, download=False, samples_per_class=None,
                 class_indices=None, pred_vs_true=None, seed_save_dir=None, seed_id=0):
        """
        data   : images as numpy.array to not load the entire dataset in GPU
        targets: labels as torch.tensor
        seed_save_dir is None if a subset of training dataset is to be created
        """
        self.dataset_name = dataset_name
        seed_path = f"{seed_save_dir}/{dataset_name}/{dataset_name}_seedset_{seed_id}.pickle"
        if seed_save_dir is not None and os.path.exists(seed_path):
            print("loading seed set from ", seed_path)
            with open(seed_path, "rb") as f:
                loaded_data = pickle.load(f)
                self.data = loaded_data['data']
                self.targets = loaded_data['targets']
                self.selected_idx = loaded_data['selected_idx']
            print("Saved data loaded")

        else:
            assert dataset_name in ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST"], "dataset_name should be MNIST/CIFAR10/CIFAR100/SVHN/FashionMNIST"

            # Only for MNIST dataset, data and targets are already an instance of torch.Tensor
            if dataset_name == "MNIST" or dataset_name == "FashionMNIST":
                dataset = eval(dataset_name)(root=root, train=train, download=download)
                data = dataset.data.numpy()
                targets = dataset.targets.long().numpy()

            elif dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
                dataset = eval(dataset_name)(root=root, train=train, download=download)
                data = dataset.data			    # type(data) = <class 'numpy.ndarray'>
                targets = dataset.targets       # type(targets) = <class 'list'>
                targets = np.array(targets)     # type(targets) = <class 'numpy.ndarray'>
            elif dataset_name == "SVHN":
                split = "train" if train else "test"
                dataset = SVHN(root=root+"/svhn", split=split, download=download)

                # reshape data from [_, 3, 32, 32] to [_, 32, 32, 3] for loading as an image
                data = dataset.data.transpose((0, 2, 3, 1))
                targets = dataset.labels

            if dataset_name in ["CIFAR100", "ImageNet"]:
                self.n_class = 100
            elif dataset_name in ["MNIST", "SVHN", "FashionMNIST", "CIFAR10"]:
                self.n_class = 10

            self.selected_idx = []

            # images which are correctly classified should be used for seed set
            # Important! Dont remove this if condition else the order of data will change and pred_vs_true will not be valid
            if samples_per_class is None:
                self.selected_idx = np.arange(targets.shape[0]) 
            else:
                print("Data order will be changed")

                # only classes in class_indices will be considered
                if class_indices is None:
                    class_indices = [i for i in range(self.n_class)]

                if pred_vs_true is None:
                    pred_vs_true = (np.arange(targets.shape[0])>0)

                all_idx = np.arange(targets.shape[0])            
                for i in class_indices:
                    idx = all_idx[np.logical_and(targets == i, pred_vs_true)]
                    # print(i, idx.shape, np.sum(targets[idx]), np.sum(pred_vs_true[idx]))
                    # only #samples_per_class to be used for each class
                    if samples_per_class is not None:
                        idx = np.random.choice(idx, samples_per_class, replace=False)
                    # print(i, idx.shape, np.sum(targets[idx]), np.sum(pred_vs_true[idx]))
                    self.selected_idx.append(copy.deepcopy(idx))

                self.selected_idx = np.concatenate(self.selected_idx, axis=0) # self.selected_idx.shape = (1000,)
            # print(np.sum(pred_vs_true[self.selected_idx]))
            self.data = data[self.selected_idx]
            self.targets = torch.from_numpy(targets[self.selected_idx])

            # SAVING SEED SET FOR REPRODUCIBILITY
            if seed_save_dir is not None:
                seed_path = f"{seed_save_dir}/{dataset_name}"
                os.makedirs(seed_path, exist_ok=True)
                seed_path = f"{seed_path}/{dataset_name}_seedset_{seed_id}.pickle"
                with open(seed_path, "wb") as f:
                    pickle.dump({'data': self.data, "targets": self.targets,
                                "selected_idx": self.selected_idx, "pred_vs_true": pred_vs_true}, f)
                print(f"Seed set saved to : {seed_path}")

        # print("Size of dataset: ", self.data.shape, self.targets.shape)
        # print("class distribution in dataset", np.unique(self.targets, return_counts=True))

        self.transform = transforms.Compose([transforms.ToTensor()]) #image_transforms[dataset_name]

        self.normalise_imgs = normalise_imgs
        if "CIFAR" in dataset_name:
            self.normalise = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            self.normalise = None  # MNIST

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        if self.dataset_name != 'MNIST':
            img = Image.fromarray(img)

        img = self.transform(img)

        if self.normalise_imgs and self.normalise is not None:
            img = self.normalise(img)

        # if self.random_transforms:
        # 	tidx = np.random.choice(self.n_transforms, size=1, replace=False)[0]
        # 	random_transformed_img = self.transform_list[tidx](copy.deepcopy(img))
        # 	return ((index, img, random_transformed_img), target)

        return (img, target)

    def __len__(self):
        return len(self.data)

    def build_and_shuffle(self):
        image_list = []
        label_list = []
        n_inputs = self.__len__()
        randomize_idx = torch.randperm(n_inputs)
        for i in randomize_idx:
            (image, label) = self.__getitem__(i)
            image_list.append(image)
            label_list.append(label)

        return image_list, label_list


def test_code_sanity():
    """
    Iterating over the dataset object returns the items sequentially
    """
    seed_set = Data(dataset_name='CIFAR10',
                    root='../../data',
                    train=False,
                    download=True,
                    class_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    samples_per_class=100,
                    random_transforms=False)

    for img_id, (nat_img, true_label) in seed_set:
        print(img_id, nat_img.shape, true_label)

# test_code_sanity()
