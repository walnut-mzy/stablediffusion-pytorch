import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os


class MNISTDataset:

    def __init__(
        self,
        root_dir="./data",
        image_size=32,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        download=True,
        use_augmentation=False,
        normalize_to_neg_one_to_one=True,
        convert_to_rgb=True,
        data_percentage=100.0
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.use_augmentation = use_augmentation
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.convert_to_rgb = convert_to_rgb
        self.data_percentage = max(0.1, min(100.0, data_percentage))  # 限制在0.1%到100%之间
        
        os.makedirs(root_dir, exist_ok=True)
        
        self.setup_transforms()
        
        self.load_datasets()
        
        self.create_dataloaders()
        
        channels = 3 if self.convert_to_rgb else 1
        print(f"MNIST数据集加载完成:")
        print(f"  - 数据使用比例: {self.data_percentage:.1f}%")
        print(f"  - 训练集大小: {len(self.train_dataset)}")
        print(f"  - 测试集大小: {len(self.test_dataset)}")
        print(f"  - 图像尺寸: {self.image_size}x{self.image_size}")
        print(f"  - 图像通道数: {channels} ({'RGB' if self.convert_to_rgb else '灰度'})")
        print(f"  - 批大小: {self.batch_size}")
        
    def setup_transforms(self):
        transform_list = []
        
        if self.image_size != 28:
            transform_list.append(transforms.Resize(self.image_size))
        
        transform_list.append(transforms.ToTensor())
        
        if self.convert_to_rgb:
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
        if self.normalize_to_neg_one_to_one:
            if self.convert_to_rgb:
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            if self.convert_to_rgb:
                transform_list.append(transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)))
            else:
                transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        self.base_transform = transforms.Compose(transform_list)
        
        if self.use_augmentation:
            augment_list = [
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ]
            
            if self.image_size != 28:
                augment_list.append(transforms.Resize(self.image_size))
            
            augment_list.append(transforms.ToTensor())
            
            if self.convert_to_rgb:
                augment_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            
            if self.normalize_to_neg_one_to_one:
                if self.convert_to_rgb:
                    augment_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
                else:
                    augment_list.append(transforms.Normalize((0.5,), (0.5,)))
            else:
                if self.convert_to_rgb:
                    augment_list.append(transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)))
                else:
                    augment_list.append(transforms.Normalize((0.1307,), (0.3081,)))
            
            self.train_transform = transforms.Compose(augment_list)
        else:
            self.train_transform = self.base_transform
        
        self.test_transform = self.base_transform
    
    def load_datasets(self):
        # 首先加载完整数据集
        full_train_dataset = torchvision.datasets.MNIST(
            root=self.root_dir,
            train=True,
            transform=self.train_transform,
            download=self.download
        )
        
        full_test_dataset = torchvision.datasets.MNIST(
            root=self.root_dir,
            train=False,
            transform=self.test_transform,
            download=self.download
        )
        
        # 如果使用部分数据，则进行子采样
        if self.data_percentage < 100.0:
            self.train_dataset = self._subsample_dataset(full_train_dataset, self.data_percentage)
            self.test_dataset = self._subsample_dataset(full_test_dataset, self.data_percentage)
        else:
            self.train_dataset = full_train_dataset
            self.test_dataset = full_test_dataset
    
    def _subsample_dataset(self, dataset, percentage):
        """按百分比子采样数据集，保持类别平衡"""
        from torch.utils.data import Subset
        import random
        
        # 计算需要保留的样本数量
        total_samples = len(dataset)
        keep_samples = int(total_samples * percentage / 100.0)
        keep_samples = max(1, keep_samples)  # 至少保留1个样本
        
        # 获取所有样本的标签
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]
        
        targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
        
        # 为每个类别收集索引
        class_indices = {}
        for idx, label in enumerate(targets):
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 计算每个类别应该保留的样本数
        num_classes = len(class_indices)
        samples_per_class = keep_samples // num_classes
        remaining_samples = keep_samples % num_classes
        
        selected_indices = []
        
        # 为每个类别随机选择样本
        for i, (label, indices) in enumerate(class_indices.items()):
            # 前remaining_samples个类别多选择1个样本
            current_samples = samples_per_class + (1 if i < remaining_samples else 0)
            current_samples = min(current_samples, len(indices))  # 不超过该类别的总样本数
            
            if current_samples > 0:
                random.shuffle(indices)
                selected_indices.extend(indices[:current_samples])
        
        # 创建子集
        random.shuffle(selected_indices)  # 打乱顺序
        subset = Subset(dataset, selected_indices)
        
        print(f"    子采样: {total_samples} -> {len(subset)} 样本 ({percentage:.1f}%)")
        
        return subset
    
    def create_dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_sample_batch(self, from_train=True):
        loader = self.train_loader if from_train else self.test_loader
        return next(iter(loader))
    
    def denormalize(self, tensor):
        if self.normalize_to_neg_one_to_one:
            return (tensor + 1.0) / 2.0
        else:
            if self.convert_to_rgb:
                mean = torch.tensor([0.1307, 0.1307, 0.1307]).view(1, 3, 1, 1)
                std = torch.tensor([0.3081, 0.3081, 0.3081]).view(1, 3, 1, 1)
                if tensor.device != mean.device:
                    mean = mean.to(tensor.device)
                    std = std.to(tensor.device)
                return tensor * std + mean
            else:
                return tensor * 0.3081 + 0.1307
    
    def convert_to_rgb(self, images):
        if self.convert_to_rgb:
            return images
        else:
            return images.repeat(1, 3, 1, 1)
    
    def get_class_names(self):
        return [str(i) for i in range(10)]
    
    def get_dataset_info(self):
        channels = 3 if self.convert_to_rgb else 1
        return {
            "num_classes": 10,
            "num_channels": channels,
            "image_size": self.image_size,
            "train_size": len(self.train_dataset),
            "test_size": len(self.test_dataset),
            "class_names": self.get_class_names(),
            "normalization": "[-1, 1]" if self.normalize_to_neg_one_to_one else "[0, 1]",
            "convert_to_rgb": self.convert_to_rgb,
            "use_augmentation": self.use_augmentation
        }
    
    def visualize_samples(self, num_samples=8, from_train=True, save_path=None):
        import matplotlib.pyplot as plt
        
        images, labels = self.get_sample_batch(from_train)
        images = self.denormalize(images)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            if self.convert_to_rgb:
                img = images[i].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
            else:
                img = images[i].squeeze().numpy()
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {labels[i].item()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"样本可视化已保存到: {save_path}")
        else:
            plt.show()

class ImageNetDataset:
    """
    ImageNet数据集处理类
    支持64x64, 128x128, 256x256等分辨率
    """
    
    def __init__(
        self,
        root_dir="./data/imagenet",
        image_size=64,  # ImageNet通常使用64x64或更高分辨率
        batch_size=32,  # ImageNet需要较小的batch size
        num_workers=8,
        pin_memory=True,
        download=False,  # ImageNet通常需要手动下载
        use_augmentation=True,
        normalize_to_neg_one_to_one=True,
        data_percentage=100.0,
        subset_classes=None,  # 可以指定使用部分类别进行训练
        center_crop=True
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.use_augmentation = use_augmentation
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.data_percentage = max(0.1, min(100.0, data_percentage))
        self.subset_classes = subset_classes
        self.center_crop = center_crop
        
        # ImageNet的标准统计信息
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        os.makedirs(root_dir, exist_ok=True)
        
        self.setup_transforms()
        
        self.load_datasets()
        
        self.create_dataloaders()
        
        self.num_classes = 1000 if subset_classes is None else len(subset_classes)
        
        print(f"ImageNet数据集加载完成:")
        print(f"  - 数据使用比例: {self.data_percentage:.1f}%")
        print(f"  - 训练集大小: {len(self.train_dataset)}")
        print(f"  - 验证集大小: {len(self.test_dataset)}")
        print(f"  - 图像尺寸: {self.image_size}x{self.image_size}")
        print(f"  - 图像通道数: 3 (RGB)")
        print(f"  - 类别数: {self.num_classes}")
        print(f"  - 批大小: {self.batch_size}")
        
    def setup_transforms(self):
        """设置数据变换"""
        # 基础变换列表
        transform_list = []
        
        # 调整大小策略
        if self.center_crop:
            # 先resize到稍大的尺寸，然后center crop
            resize_size = int(self.image_size * 1.125)  # 约1.125倍
            transform_list.extend([
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.image_size)
            ])
        else:
            # 直接resize
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        transform_list.append(transforms.ToTensor())
        
        # 归一化策略
        if self.normalize_to_neg_one_to_one:
            # 归一化到[-1, 1]，适合diffusion模型
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            # 使用ImageNet标准归一化
            transform_list.append(transforms.Normalize(self.imagenet_mean, self.imagenet_std))
        
        self.base_transform = transforms.Compose(transform_list)
        
        # 训练时的数据增强
        if self.use_augmentation:
            augment_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1, 
                    contrast=0.1, 
                    saturation=0.1, 
                    hue=0.05
                ),
            ]
            
            # 调整大小策略
            if self.center_crop:
                resize_size = int(self.image_size * 1.125)
                augment_list.extend([
                    transforms.Resize(resize_size),
                    transforms.RandomCrop(self.image_size)
                ])
            else:
                augment_list.append(transforms.Resize((self.image_size, self.image_size)))
            
            augment_list.append(transforms.ToTensor())
            
            # 归一化
            if self.normalize_to_neg_one_to_one:
                augment_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                augment_list.append(transforms.Normalize(self.imagenet_mean, self.imagenet_std))
            
            self.train_transform = transforms.Compose(augment_list)
        else:
            self.train_transform = self.base_transform
        
        self.test_transform = self.base_transform
    
    def load_datasets(self):
        """加载ImageNet数据集"""
        try:
            # 尝试加载完整ImageNet数据集
            full_train_dataset = torchvision.datasets.ImageNet(
                root=self.root_dir,
                split='train',
                transform=self.train_transform
            )
            
            full_val_dataset = torchvision.datasets.ImageNet(
                root=self.root_dir,
                split='val',
                transform=self.test_transform
            )
            
            print("成功加载完整ImageNet数据集")
            
        except Exception as e:
            print(f"无法加载完整ImageNet数据集: {e}")
            print("尝试使用ImageFolder加载自定义ImageNet数据...")
            
            # 使用ImageFolder加载自定义格式的ImageNet数据
            train_dir = os.path.join(self.root_dir, 'train')
            val_dir = os.path.join(self.root_dir, 'val')
            
            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                raise ValueError(f"请确保ImageNet数据位于 {train_dir} 和 {val_dir}")
            
            full_train_dataset = torchvision.datasets.ImageFolder(
                root=train_dir,
                transform=self.train_transform
            )
            
            full_val_dataset = torchvision.datasets.ImageFolder(
                root=val_dir,
                transform=self.test_transform
            )
            
            print("成功使用ImageFolder加载ImageNet数据")
        
        # 如果指定了子集类别，则过滤数据集
        if self.subset_classes is not None:
            full_train_dataset = self._filter_by_classes(full_train_dataset, self.subset_classes)
            full_val_dataset = self._filter_by_classes(full_val_dataset, self.subset_classes)
            print(f"已过滤数据集，使用{len(self.subset_classes)}个类别")
        
        # 如果使用部分数据，则进行子采样
        if self.data_percentage < 100.0:
            self.train_dataset = self._subsample_dataset(full_train_dataset, self.data_percentage)
            self.test_dataset = self._subsample_dataset(full_val_dataset, self.data_percentage)
        else:
            self.train_dataset = full_train_dataset
            self.test_dataset = full_val_dataset
    
    def _filter_by_classes(self, dataset, target_classes):
        """按指定类别过滤数据集"""
        from torch.utils.data import Subset
        
        # 获取所有样本的标签
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'imgs'):
            targets = [item[1] for item in dataset.imgs]
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]
        
        # 找到属于目标类别的样本索引
        target_indices = []
        for idx, label in enumerate(targets):
            if label in target_classes:
                target_indices.append(idx)
        
        return Subset(dataset, target_indices)
    
    def _subsample_dataset(self, dataset, percentage):
        """按百分比子采样数据集，保持类别平衡"""
        from torch.utils.data import Subset
        import random
        
        # 计算需要保留的样本数量
        total_samples = len(dataset)
        keep_samples = int(total_samples * percentage / 100.0)
        keep_samples = max(1, keep_samples)
        
        # 获取所有样本的标签
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset.dataset, 'targets'):  # 如果是Subset
            targets = [dataset.dataset.targets[i] for i in dataset.indices]
        elif hasattr(dataset, 'imgs'):
            targets = [item[1] for item in dataset.imgs]
        elif hasattr(dataset.dataset, 'imgs'):  # 如果是Subset of ImageFolder
            targets = [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]
        
        targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
        
        # 为每个类别收集索引
        class_indices = {}
        for idx, label in enumerate(targets):
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 计算每个类别应该保留的样本数
        num_classes = len(class_indices)
        samples_per_class = keep_samples // num_classes
        remaining_samples = keep_samples % num_classes
        
        selected_indices = []
        
        # 为每个类别随机选择样本
        for i, (label, indices) in enumerate(class_indices.items()):
            current_samples = samples_per_class + (1 if i < remaining_samples else 0)
            current_samples = min(current_samples, len(indices))
            
            if current_samples > 0:
                random.shuffle(indices)
                selected_indices.extend(indices[:current_samples])
        
        random.shuffle(selected_indices)
        subset = Subset(dataset, selected_indices)
        
        print(f"    子采样: {total_samples} -> {len(subset)} 样本 ({percentage:.1f}%)")
        
        return subset
    
    def create_dataloaders(self):
        """创建数据加载器"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_sample_batch(self, from_train=True):
        loader = self.train_loader if from_train else self.test_loader
        return next(iter(loader))
    
    def denormalize(self, tensor):
        """反归一化图像tensor"""
        if self.normalize_to_neg_one_to_one:
            return (tensor + 1.0) / 2.0
        else:
            # 使用ImageNet标准统计信息反归一化
            mean = torch.tensor(self.imagenet_mean).view(1, 3, 1, 1)
            std = torch.tensor(self.imagenet_std).view(1, 3, 1, 1)
            if tensor.device != mean.device:
                mean = mean.to(tensor.device)
                std = std.to(tensor.device)
            return tensor * std + mean
    
    def get_class_names(self):
        """获取类别名称"""
        if self.subset_classes is not None:
            return [f"class_{i}" for i in self.subset_classes]
        else:
            # ImageNet有1000个类别
            return [f"class_{i}" for i in range(1000)]
    
    def get_dataset_info(self):
        return {
            "train_size": len(self.train_dataset),
            "test_size": len(self.test_dataset),
            "num_classes": self.num_classes,
            "normalization": "[-1, 1]" if self.normalize_to_neg_one_to_one else "ImageNet Stats",
            "data_percentage": self.data_percentage,
            "subset_classes": self.subset_classes,
            "use_augmentation": self.use_augmentation
        }
    
    def visualize_samples(self, num_samples=8, from_train=True, save_path=None):
        """可视化样本"""
        import matplotlib.pyplot as plt
        
        images, labels = self.get_sample_batch(from_train)
        images = self.denormalize(images)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            img = images[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].set_title(f'Class: {labels[i].item()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"样本可视化已保存到: {save_path}")
        else:
            plt.show()

class CIFAR10Dataset:
    """
    CIFAR-10数据集处理类
    """
    
    def __init__(
        self,
        root_dir="./data/cifar10",
        image_size=32,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        download=True,
        use_augmentation=True,
        normalize_to_neg_one_to_one=True,
        data_percentage=100.0
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.use_augmentation = use_augmentation
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.data_percentage = max(0.1, min(100.0, data_percentage))
        
        os.makedirs(root_dir, exist_ok=True)
        
        self.setup_transforms()
        
        self.load_datasets()
        
        self.create_dataloaders()
        
        print(f"CIFAR-10数据集加载完成:")
        print(f"  - 数据使用比例: {self.data_percentage:.1f}%")
        print(f"  - 训练集大小: {len(self.train_dataset)}")
        print(f"  - 测试集大小: {len(self.test_dataset)}")
        print(f"  - 图像尺寸: {self.image_size}x{self.image_size}")
        print(f"  - 图像通道数: 3 (RGB)")
        print(f"  - 批大小: {self.batch_size}")
        
    def setup_transforms(self):
        transform_list = []
        
        if self.image_size != 32:
            transform_list.append(transforms.Resize(self.image_size))
        
        transform_list.append(transforms.ToTensor())
        
        if self.normalize_to_neg_one_to_one:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            # CIFAR-10的标准统计信息
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        
        self.base_transform = transforms.Compose(transform_list)
        
        if self.use_augmentation:
            augment_list = [
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            
            if self.image_size != 32:
                augment_list.append(transforms.Resize(self.image_size))
            
            augment_list.append(transforms.ToTensor())
            
            if self.normalize_to_neg_one_to_one:
                augment_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                augment_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
            
            self.train_transform = transforms.Compose(augment_list)
        else:
            self.train_transform = self.base_transform
        
        self.test_transform = self.base_transform

    def load_datasets(self):
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.root_dir,
            train=True,
            transform=self.train_transform,
            download=self.download
        )
        
        full_test_dataset = torchvision.datasets.CIFAR10(
            root=self.root_dir,
            train=False,
            transform=self.test_transform,
            download=self.download
        )
        
        if self.data_percentage < 100.0:
            self.train_dataset = self._subsample_dataset(full_train_dataset, self.data_percentage)
            self.test_dataset = self._subsample_dataset(full_test_dataset, self.data_percentage)
        else:
            self.train_dataset = full_train_dataset
            self.test_dataset = full_test_dataset

    def _subsample_dataset(self, dataset, percentage):
        from torch.utils.data import Subset
        import random
        
        total_samples = len(dataset)
        keep_samples = int(total_samples * percentage / 100.0)
        
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        subset_indices = indices[:keep_samples]
        
        subset = Subset(dataset, subset_indices)
        
        print(f"    子采样: {total_samples} -> {len(subset)} 样本 ({percentage:.1f}%)")
        return subset

    def create_dataloaders(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def get_sample_batch(self, from_train=True):
        loader = self.train_loader if from_train else self.test_loader
        return next(iter(loader))

    def denormalize(self, tensor):
        if self.normalize_to_neg_one_to_one:
            return (tensor + 1.0) / 2.0
        else:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
            if tensor.device != mean.device:
                mean = mean.to(tensor.device)
                std = std.to(tensor.device)
            return tensor * std + mean
            
    def get_class_names(self):
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def get_dataset_info(self):
        return {
            "num_classes": 10,
            "num_channels": 3,
            "image_size": self.image_size,
            "train_size": len(self.train_dataset),
            "test_size": len(self.test_dataset),
            "class_names": self.get_class_names(),
            "normalization": "[-1, 1]" if self.normalize_to_neg_one_to_one else "[0, 1]",
            "use_augmentation": self.use_augmentation
        }

    def visualize_samples(self, num_samples=8, from_train=True, save_path=None):
        import matplotlib.pyplot as plt
        
        images, labels = self.get_sample_batch(from_train)
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        denorm_images = self.denormalize(images)
        
        plt.figure(figsize=(16, 4))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            img = denorm_images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(self.get_class_names()[labels[i]])
            plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            print(f"样本图像已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close() 