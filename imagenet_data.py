from torchvision import transforms
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import pickle
import os
import json

image_transforms = {
    "ImageNet": transforms.Compose([transforms.ToTensor()])
}


class ImageNet_Data:
    """
    =====================================================================================
    Customised dataloader class for ImageNet dataset

    samples_per_class: for reduced training set with #images_per_class = samples_per_class
    num_class        : number of classes to be considered
    ======================================================================================
    """

    def __init__(self, image_dir, label2index_file, split, normalise_imgs, download=False, num_class=None, samples_per_class=None,
                 pred_vs_true=None, seed_save_dir=None, image_size=128, seed_id=0):
        """
        data: list of image names
        targets: labels as torch.tensor
        """
        if seed_save_dir is not None:
            seed_path = f"{seed_save_dir}/ImageNet"

        if seed_save_dir is not None and os.path.exists(f"{seed_path}/ImageNet_seedset_{seed_id}.pickle"):
            with open(f"{seed_path}/ImageNet_seedset_{seed_id}.pickle", "rb") as f:
                loaded_data = pickle.load(f)
                self.image_list = loaded_data['image_list']
            print("Saved data loaded")

        else:
            image_dir = image_dir + split + ('/' if len(split) else '')
            class_list = sorted(os.listdir(image_dir))

            self.image_list = []
            for class_name in class_list:
                name_list = sorted(os.listdir(image_dir + class_name))
                self.image_list += [image_dir + class_name + '/' + image_name for image_name in name_list]

            if pred_vs_true is not None:
                self.image_list = np.array(self.image_list)
                self.image_list = self.image_list[np.arange(len(self.image_list))[pred_vs_true]]

            if num_class is not None:     
                new_image_list = []
                for class_name in class_list[:num_class]:
                    cls_name_list = [image_dir + class_name + '/' + i for i in os.listdir(image_dir + class_name)]
                    # print("class_name", class_name)
                    # print("Initial: len(cls_name_list)", len(cls_name_list))
                    selected_names = np.intersect1d(np.array(cls_name_list), self.image_list)
                    # print("Intersection: selected_names.shape, self.image_list.shape", selected_names.shape, self.image_list.shape)
                    
                    if samples_per_class is not None:
                        selected_names = np.random.choice(selected_names, size=min(len(selected_names), samples_per_class), replace=False)
                        # print("samples_per_class: selected_names.shape, self.image_list.shape", selected_names.shape, self.image_list.shape)
                
                    new_image_list.append(selected_names)
                new_image_list = np.concatenate(new_image_list, axis=0)
                # print("new_image_list.shape", new_image_list.shape)
                    
                # if size of seed-set is less than 1000
                diff = samples_per_class*num_class - len(new_image_list)
                if diff>0:
                    more_images = []
                    for name in self.image_list:
                        if name not in new_image_list:
                            more_images.append(name)
                        
                        if len(more_images) == diff:
                            break
                    # print(len(more_images), new_image_list.shape)
                    new_image_list = np.concatenate((new_image_list, np.array(more_images)), axis=0)
                    # print(new_image_list.shape)

                self.image_list = new_image_list
                # print(self.image_list.shape)
            # exit()

            # SAVING SEED SET FOR REPRODUCIBILITY
            if seed_save_dir is not None:
                os.makedirs(seed_path, exist_ok=True)
                with open(f"{seed_path}/ImageNet_seedset_{seed_id}.pickle", "wb") as f:
                    pickle.dump({'image_list': self.image_list}, f)                                
                print(f"Seed set saved to : {seed_path}/ImageNet_seedset_{seed_id}.pickle")
            print("Size of dataset: ", len(self.image_list))

        self.transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor()
                    ])
        self.normalise_imgs = normalise_imgs
        self.normalise = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    
        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        
    def __getitem__(self, index):

        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name
        # print(image_path, label)
        label = self.label2index[label]
        label = torch.LongTensor([label]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.normalise_imgs:
            image = self.normalise(image)

        return image, label

        # if self.random_transforms:
        # 	tidx = np.random.choice(self.n_transforms, size=1, replace=False)[0]
        # 	random_transformed_img = self.transform_list[tidx](copy.deepcopy(img))
        # 	return ((index, img, random_transformed_img), target)


    def __len__(self):
        return len(self.image_list)

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

    # def to_numpy(self, image_list, is_image=True):
    #     image_numpy_list = []
    #     for i in range(len(image_list)):
    #         image = image_list[i]  # torch.Tensor
    #         if is_image:
    #             image_numpy = image.transpose(0, 2).numpy()
    #         else:
    #             image_numpy = image.numpy()
    #         image_numpy_list.append(image_numpy)
    #     return image_numpy_list


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
