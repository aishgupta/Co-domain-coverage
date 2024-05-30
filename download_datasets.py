import torchvision

mnist = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=None)
mnist = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=None)

fashionmnist = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=None)
fashionmnist = torchvision.datasets.FashionMNIST(root='./data/', train=False, download=True, transform=None)

svhn = torchvision.datasets.SVHN(root='./data/', split='train', download=True, transform=None)
svhn = torchvision.datasets.SVHN(root='./data/', split='test', download=True, transform=None)

cifar10 = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=None)
cifar10 = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=None)

cifar100 = torchvision.datasets.CIFAR100(root='./data/', train=True, download=True, transform=None)
cifar100 = torchvision.datasets.CIFAR100(root='./data/', train=False, download=True, transform=None)