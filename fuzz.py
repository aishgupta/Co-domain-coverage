import sys
import copy
import random
import numpy as np
import time
from tqdm import tqdm
import itertools
import gc
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image

import coverage
import utility

from PIL import Image


class Parameters(object):
    def __init__(self, base_args):
        self.model = base_args.model
        self.dataset = base_args.dataset
        self.criterion = base_args.criterion
        self.only_last = base_args.only_last
        self.max_testsuite_size = base_args.max_testsuite_size
        self.seed_id = base_args.seed_id

        self.use_sc = self.criterion in ['LSC', 'DSC', 'MDSC']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 2

        self.batch_size = 128
        self.mutate_batch_size = 1

        self.image_size = constants.data_info[self.dataset]['img_size']
        self.nc = constants.data_info[self.dataset]['n_img_channels']
        self.n_test_class = constants.data_info[self.dataset]['n_test_class']
        self.input_shape = (1, self.image_size, self.image_size, self.nc)
        self.num_per_class = 1000 // self.n_test_class

        if self.criterion != 'random':
            self.criterion_hp = constants.hyper_map[self.criterion]
        
        self.noise_data = False
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        self.alpha = 0.2 
        self.beta = 0.4  
        self.TRY_NUM = 50
        self.save_every = 100
        self.output_dir = base_args.output_dir

        self.save_batch = False


class INFO(object):
    def __init__(self):
        self.dict = {}
        self.transformlist = []

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, state = i, 0
            return I0, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]


class Fuzzer:
    def __init__(self, params, criterion):
        self.params = params
        self.epoch = 0
        self.time_slot = 60 * 10
        self.time_idx = 0
        self.info = INFO()
        self.hyper_params = {
            'alpha': 0.2,  
            'beta': 0.4,  
            'TRY_NUM': 50,
            'p_min': 0.01,
            'gamma': 5,
            'K': 64
        }
        self.logger = utility.Logger(params, self)
        self.criterion = criterion
        if criterion != 'random':
            self.initial_coverage = copy.deepcopy(criterion.current)
        else:
            self.initial_coverage = 0
        self.delta_time = 0
        self.coverage_time = 0
        self.delta_batch = 0
        self.num_ae = 0

        self.transform_list = constants.dataset_transforms[self.params.dataset]
        self.n_transforms = len(self.transform_list["pixel"]) + len(self.transform_list["affine"])

        self.testsuite = []

    def exit(self):
        self.print_info()
        if self.criterion!='random':
            self.criterion.save(self.params.coverage_dir + 'coverage.pt')
        # self.logger.save()
        self.logger.exit()

        self.testsuite.append((self.coverage_time, self.delta_time))
        with open(f'{self.params.coverage_dir}testsuite.pkl', "wb") as f:
            pickle.dump(self.testsuite, f)

    def can_terminate(self):
        condition = sum([
            self.epoch > 10000,
            self.delta_time > 60 * 60 * 6,
            len(self.testsuite) > self.params.max_testsuite_size,
        ])
        return condition > 0

    def print_info(self):
        self.logger.update(self)

    def is_adversarial(self, image, label, k=1):
        """
        image will be given as input to the model
        """
        with torch.no_grad():
            scores = self.criterion.model(image)
            _, ind = scores.topk(k, dim=1, largest=True, sorted=True)

            correct = ind.eq(label.view(-1, 1).expand_as(ind))
            wrong = ~correct
            index = (wrong == True).nonzero(as_tuple=True)[0]
            wrong_total = wrong.view(-1).float().sum()

            return wrong_total, index

    def run(self, I_input, L_input):
        """
        B is a list containing an image as a numpy array of shape (32, 32, 3)
        B_label is a list containing a scalar int as a label
        B_id is the index of the selected seed
        """
        selection_priority = (list(np.zeros(len(I_input))))

        self.epoch = 0
        start_time = time.time()
        selected_seed_id = self.SelectNext(selection_priority)
        while not self.can_terminate():
            if self.epoch % 500 == 0:
                self.print_info()
            
            seed_img = I_input[selected_seed_id]
            label = L_input[selected_seed_id]
            Ps = self.PowerSchedule([seed_img], self.hyper_params['K'])
            img_gen = False

            time_per_seed = 0
            for i in range(1, Ps(0) + 1):
                I_new = self.Mutate(seed_img)
                if self.isFailedTest(I_new):
                    F += np.concatenate((F, [I_new.cpu().numpy()]))
                elif self.isChanged(seed_img, I_new):
                    if seed_set.normalise is not None:
                        I_new_norm = seed_set.normalise(I_new)
                    else:
                        I_new_norm = I_new

                    I_new_norm = I_new_norm.to(self.params.device).unsqueeze(0)
                    label = label.to(self.params.device).unsqueeze(0)

                    # print(I_new_norm.shape)
                    if self.criterion != 'random':
                        cov_start_time = time.time()
                        if self.params.use_sc or self.params.criterion=='CDC_v2':
                            cov_dict = self.criterion.calculate(I_new_norm, label)
                        else:
                            cov_dict = self.criterion.calculate(I_new_norm)
                        gain = self.criterion.gain(cov_dict)

                        if self.CoverageGain(gain):
                            self.criterion.update(cov_dict, gain)
                            img_gen = True
                            # B_new = np.concatenate((B_new, [I_new]))
                            # B_old = np.concatenate((B_old, [I]))
                            # B_label_new += [L]
                        cov_end_time = time.time()
                        time_per_seed += (cov_end_time - cov_start_time)
                    else:
                        img_gen = True

                    if img_gen == True:
                        break

            if img_gen:
                I_input.append(I_new)
                L_input.append(label.cpu().squeeze())
                selection_priority.append(selection_priority[selected_seed_id] * 0)
                selection_priority[selected_seed_id] += 1
                self.delta_batch += 1

                self.testsuite.append((I_new, label))

            gc.collect()

            selected_seed_id = self.SelectNext(selection_priority)
            self.epoch += 1     # number of seeds selected before exiting
            self.coverage_time += time_per_seed
            self.delta_time = time.time() - start_time

    def calc_priority(self, B_ci):
        if B_ci < (1 - self.hyper_params['p_min']) * self.hyper_params['gamma']:
            return 1 - B_ci / self.hyper_params['gamma']
        else:
            return self.hyper_params['p_min']

    def SelectNext(self, B_c):
        B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        c = np.random.choice(len(B_c), p=B_p / np.sum(B_p))
        return c

    def PowerSchedule(self, S, K):
        potentials = []
        for i in range(len(S)):
            I = S[i]
            I0, state = self.info[I]
            # print(I0.shape)
            p = self.hyper_params['beta'] * 255 * torch.sum(I > 0) - torch.sum(torch.abs(I - I0))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)  # potentials = [1.]

        def Ps(I_id):
            p = potentials[I_id]
            return int(np.ceil(p * K))
        return Ps

    def isFailedTest(self, I_new):
        return False

    def isChanged(self, I, I_new):
        return torch.any(I != I_new)

    def CoverageGain(self, gain):
        if gain is not None:
            if isinstance(gain, tuple):
                return gain[0] > 0
            else:
                return gain > 0
        else:
            return False

    def Mutate(self, I: torch.tensor) -> torch.tensor:
        """
        Mutate image by applying transformations. 
        affine_trans is initially False and is set to True when an affine transformation is applied. 
        However, if the affine transformed image is not valid then, affine_trans is set to False again.
        """
        I0, state = self.info[I]
        assert (I0.ndim == 3) and (I.ndim == 3), "image size is incorrect"

        affine_trans = False
        for i in range(1, self.hyper_params['TRY_NUM']):
            if state == 0:
                tidx = np.random.choice(self.n_transforms, size=1, replace=False)[0]

                # if affine transformation is selected
                if tidx >= len(self.transform_list["pixel"]):
                    affine_trans = True
                    tidx = tidx - len(self.transform_list["pixel"])
                    I_mutated = self.transform_list["affine"][tidx](I)
                    I0_G = self.transform_list["affine"][tidx](I0)
                else:
                    I_mutated = self.transform_list["pixel"][tidx](I)
            else:
                tidx = np.random.choice(len(self.transform_list["pixel"]), size=1, replace=False)[0]
                I_mutated = self.transform_list["pixel"][tidx](I)

            I_mutated = torch.clamp(I_mutated, min=0, max=1)

            if self.f(I0, I_mutated):
                if affine_trans:
                    state = 1
                    self.info[I_mutated] = (I0_G, state)
                else:
                    self.info[I_mutated] = (I0, state)
                return I_mutated
            
        return I

    def saveImage(self, image, path):
        if image is not None:
            print('Saving mutated images in %s...' % path)
            save_image(image, path)

    def f(self, I, I_new):
        l0_dist = torch.sum((I - I_new) != 0)
        linf_dist = torch.max(torch.abs(I - I_new))

        if (l0_dist < self.hyper_params['alpha'] * torch.sum(I > 0)):
            return linf_dist <= 255
        else:
            return linf_dist <= self.hyper_params['beta'] * 255


if __name__ == '__main__':
    import os
    import sys
    import argparse
    import torchvision
    import gc

    import utility
    import models
    from models.mnist_models import LeNet
    import tool
    import coverage
    import constants
    # import data_loader
    from datasets import Data
    from imagenet_data import ImageNet_Data

    import signal

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        try:
            if engine is not None:
                engine.print_info()
                engine.exit()
        except:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['lenet5', 'LeNet', 'resnet18', 'resnet34', 'resnet50', 'vgg16', 'mobilenet_v2'])
    parser.add_argument('--criterion', type=str, default='NLC',
                        choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                                 'LSC', 'DSC', 'MDSC', 'CDC', 'random', 'CDC_v2'])
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./seed_dir')
    parser.add_argument('--only_last', action='store_true')

    parser.add_argument('--max_testsuite_size', type=int, default=10000)

    parser.add_argument('--n_bins', type=int, default=100)
    parser.add_argument('--max_per_bin', type=int, default=10)

    # parser.add_argument('--hyper', type=str, default=None)
    base_args = parser.parse_args()

    if 'CDC' in base_args.criterion:
        constants.hyper_map[base_args.criterion]['n_class'] = constants.data_info[base_args.dataset]['n_class']
        constants.hyper_map[base_args.criterion]['n_bins'] = base_args.n_bins
        constants.hyper_map[base_args.criterion]['max_per_bin'] = base_args.max_per_bin

    args = Parameters(base_args)

    args.exp_name = ('%s-%s-%s' % (args.dataset, args.model, args.criterion))
    if args.criterion in ["NLC", "CC"] and args.only_last:
        args.exp_name = args.exp_name + "-only-last-layer"
    elif 'CDC' in args.criterion:
        args.exp_name = f"{args.exp_name}_{constants.hyper_map[args.criterion]['n_bins']}_{constants.hyper_map[args.criterion]['max_per_bin']}"
    elif args.criterion in ['LSC', 'DSC']:
        args.exp_name = f"{args.exp_name}_{constants.hyper_map[args.criterion]['ub']}_{constants.hyper_map[args.criterion]['n_bins']}"
    elif args.criterion != 'random':
        args.exp_name = f"{args.exp_name}_{constants.hyper_map[args.criterion]}"
    args.exp_name = f"{args.exp_name}_fixed_{args.max_testsuite_size}"

    args.image_dir = f'{args.output_dir}/{args.dataset}/{args.exp_name}/image/'
    args.coverage_dir = f'{args.output_dir}/{args.dataset}/{args.exp_name}/coverage/'
    args.log_dir = f'{args.output_dir}/{args.dataset}/{args.exp_name}/log/'

    print(args.__dict__)

    utility.make_path(args.image_dir)
    utility.make_path(args.coverage_dir)
    utility.make_path(args.log_dir)

    TOTAL_CLASS_NUM = constants.data_info[args.dataset]['n_class']
    if args.dataset == 'ImageNet':
        model = torchvision.models.__dict__[args.model](pretrained=True)
        path = None
        # path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 128
        assert args.n_test_class <= 1000
    elif 'CIFAR' in args.dataset:
        model = getattr(models, args.model)(pretrained=False, num_classes=TOTAL_CLASS_NUM)
        if 'resnet' in args.model:
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 32
        assert args.n_test_class <= 10 if args.dataset == 'CIFAR10' else args.n_test_class <= 100
    elif args.dataset == 'MNIST':
        model = getattr(models, args.model)(pretrained=False)
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pt' % (args.dataset, args.model)))
        assert args.image_size == 28
        assert args.n_test_class <= 10
        assert args.nc == 1, "number of channels incorrect"
    elif args.dataset == 'SVHN':
        # model = torchvision.models.vgg16_bn(pretrained=True)
        # model.classifier[6] = torch.nn.Linear(4096, 10, True)
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 32
        assert args.n_test_class <= 10
        assert args.nc == 3, "number of channels incorrect"
    elif args.dataset == 'FashionMNIST':
        model = LeNet()
        path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
        assert args.image_size == 28
        assert args.n_test_class <= 10
        assert args.nc == 1, "number of channels incorrect"

    # test model with pretrained weights
    # model.load_state_dict(torch.load("NeuraL-Coverage/pretrained_models/ImageNet/mobilenet_v2_pytorch_weights.pth"))
    if path is not None:
        if "CIFAR" in args.dataset and "resnet" in args.model:
            sd = torch.load(path)
            model.load_state_dict(sd["net"])
        elif args.dataset == 'SVHN':
            model = torch.load(path)
        elif args.dataset == 'FashionMNIST':
            model = torch.load(path)
        else:
            model.load_state_dict(torch.load(path))
    model.to(args.device)
    model.eval()

    if args.dataset != 'ImageNet':
        print("Evaluating model's performance on test dataset")
        print("----------------------------------------------")
        testdata = Data(dataset_name=args.dataset,
                    root="./data",
                    train=False,
                    download=True,
                    normalise_imgs=True)
        dataloader = torch.utils.data.DataLoader(
            testdata,
            batch_size=512,
            shuffle=False, drop_last=False)

        with torch.no_grad():
            correct = 0

            for idx, (img, label) in enumerate(dataloader):
                img = img.to(args.device)
                label = label.to(args.device)

                output = model(img)
                pred_label = torch.max(output, dim=1)[1]

                assert (pred_label.shape == label.shape), print("Shape mismatch.")
                
                correct += torch.sum(pred_label == label).item()

                if idx == 0:
                    pred_vs_true = (pred_label == label)
                    all_pred_label = pred_label
                else:
                    pred_vs_true = torch.cat((pred_vs_true, (pred_label == label)), dim=0)
                    all_pred_label = torch.cat((all_pred_label, pred_label), dim=0)
        print("Model performance on test dataset: {}/{} = {}\n".format(correct,
                                                                       len(dataloader.dataset), 100.0 * correct / len(dataloader.dataset)))
        pred_vs_true = pred_vs_true.cpu().numpy()
        np.save(file=f"pred_vs_true_{args.dataset}_{args.model}.npy", arr=pred_vs_true)
    else:
        pred_vs_true = np.load("pred_vs_true_imagenet_val_mobilenet_v2.npy")
        print("Model performance on test dataset: {}/{} = {}\n".format(np.sum(pred_vs_true),
                                                                       pred_vs_true.shape[0], 100.0 * np.sum(pred_vs_true) / pred_vs_true.shape[0]))

    print("Creating a seed set and evaluating model on it")
    print("----------------------------------------------")

    if args.dataset == 'ImageNet':
        seed_set = ImageNet_Data(
            image_dir=constants.IMAGENET_JPEG_DIR,
            label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
            split='val',
            download=False,
            pred_vs_true=pred_vs_true,
            num_class=args.n_test_class,
            samples_per_class=args.num_per_class,
            normalise_imgs=True,
            seed_save_dir='seed_dir',
            seed_id=args.seed_id)
    else:
        seed_set = Data(
            dataset_name=args.dataset,
            root="./data",
            train=False,
            download=True,
            samples_per_class=args.num_per_class,
            seed_save_dir='seed_dir',
            pred_vs_true=pred_vs_true,
            normalise_imgs=True,
            seed_id=args.seed_id
        )
    dataloader = torch.utils.data.DataLoader(seed_set, batch_size=512, shuffle=False, drop_last=False)

    with torch.no_grad():
        correct = 0

        for idx, (img, label) in enumerate(dataloader):
            img = img.to(args.device)
            label = label.to(args.device)

            output = model(img)
            pred_label = torch.max(output, dim=1)[1]

            assert (pred_label.shape == label.shape), print("Shape mismatch.")
            correct += torch.sum(pred_label == label).item()

            if idx == 0:
                pred_vs_true1 = (pred_label == label)
                all_pred_label = pred_label
            else:
                pred_vs_true1 = torch.cat((pred_vs_true1, (pred_label == label)), dim=0)
                all_pred_label = torch.cat((all_pred_label, pred_label), dim=0)
        pred_vs_true1 = pred_vs_true1.cpu().numpy()
    print("Model's performance on the seed set: {}/{} = {}\n".format(correct,
          len(dataloader.dataset), 100.0 * correct / len(dataloader.dataset)))
    
    print("Name of the model's layers and their output shape")
    print("-------------------------------------------------")
    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).to(args.device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)
    print(layer_size_dict)

    # recreating seed_set as normalise_imgs = False because the images have to be mutated
    print("\nCreating same seed set again with normalise_imgs=False")
    print("----------------------------------------------")
    if args.dataset == 'ImageNet':
        seed_set = ImageNet_Data(
            image_dir=constants.IMAGENET_JPEG_DIR,
            label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
            split='val',
            download=False,
            pred_vs_true=pred_vs_true,
            num_class=args.n_test_class,
            samples_per_class=args.num_per_class,
            normalise_imgs=False,
            seed_save_dir='seed_dir',
            seed_id=args.seed_id
        )
    else:
        seed_set = Data(
            dataset_name=args.dataset,
            root="./data",
            train=False,
            download=False,
            seed_save_dir='seed_dir',
            normalise_imgs=False,
            seed_id=args.seed_id
        )
    image_list, label_list = seed_set.build_and_shuffle()

    print(len(image_list), len(label_list), type(image_list[0]), type(label_list[0]), label_list[0])
    
    gc.collect()

    if args.criterion != 'random':
        if args.use_sc:
            criterion = getattr(coverage, args.criterion)(model, layer_size_dict,
                                                        hyper=constants.hyper_map[args.criterion], min_var=1e-5, num_class=TOTAL_CLASS_NUM)
        else:
            criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=constants.hyper_map[args.criterion], only_last=args.only_last)

        print("\nCreating a subset of train dataset for building and computing the initial coverage")
        print("----------------------------------------------")
        if args.dataset == 'ImageNet':
            build_set = ImageNet_Data(
                image_dir=constants.IMAGENET_JPEG_DIR,
                label2index_file=constants.IMAGENET_LABEL_TO_INDEX,
                split='val',
                download=False,
                pred_vs_true=None,
                num_class=args.n_test_class,
                samples_per_class=args.num_per_class,
                normalise_imgs=True)
        else:
            build_set = Data(dataset_name=args.dataset,
                            root="./data",
                            train=True,
                            download=True,
                            samples_per_class=args.num_per_class,
                            normalise_imgs=True
                            )

        train_loader = torch.utils.data.DataLoader(build_set, batch_size=512, shuffle=False, drop_last=False)
        criterion.build(train_loader)
        start_time = time.time()
        criterion.assess(train_loader)
        end_time = time.time()
        print("Coverage on training data: ", criterion.current, "\n")
        print("Time taken: ", end_time - start_time, "\n")

        # with open(f'coverage_training_time_{args.dataset}.txt', 'a') as f:
        #     f.write(
        #         f'${args.criterion}$ & ${constants.hyper_map[args.criterion]}$ &  ${criterion.current}$ & ${end_time-start_time}$')
        #     f.write("\n")

        '''
        # coverage cannot be initialised from the training data
        # if args.criterion not in ['LSC', 'DSC', 'MDSC']:

        For LSC/DSC/MDSC/CC/TKNP, initialization with training data is too slow (sometimes may
        exceed the memory limit). You can skip this step to speed up the experiment, which
        will not affect the conclusion because we only compare the relative order of coverage
        values, rather than the exact numbers.
        '''
        initial_coverage = copy.deepcopy(criterion.current)
        print('Initial Coverage: %f' % initial_coverage, "\n")

        if args.criterion in ['LSC', 'DSC']:
            print(criterion.train_upper_bound)
    else:
        criterion = 'random'

    engine = Fuzzer(args, criterion)
    engine.run(image_list, label_list)
    engine.exit()
