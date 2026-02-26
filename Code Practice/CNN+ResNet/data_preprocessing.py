import os
import shutil
import random
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 动物类别（选择5种常见动物）
ANIMAL_CLASSES = ['dog', 'cat', 'horse', 'chicken', 'elephant']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ANIMAL_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

class AnimalDataset(Dataset):
    """动物图像数据集类"""
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 收集所有图像路径和标签
        for class_name in ANIMAL_CLASSES:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 类别目录不存在: {class_dir}")
                continue
                
            class_images = []
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    # 验证图像是否可以打开
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        class_images.append(img_path)
                    except:
                        print(f"警告: 无法打开图像: {img_path}")
                        continue
            
            if not class_images:
                print(f"警告: 类别 {class_name} 中没有有效图像")
                continue
            
            # 分割训练集和测试集
            random.shuffle(class_images)
            split_idx = int(len(class_images) * train_ratio)
            if split == 'train':
                selected_images = class_images[:split_idx]
            else:  # test
                selected_images = class_images[split_idx:]
            for img_path in selected_images:
                self.images.append(img_path)
                self.labels.append(CLASS_TO_IDX[class_name])
        
        print(f"{split}集加载完成: {len(self.images)} 张图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
            
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}: {e}")
            # 返回一个占位图像
            placeholder = Image.new('RGB', (128, 128), color='gray')
            if self.transform:
                placeholder = self.transform(placeholder)
            return placeholder, label

def prepare_data(data_dir='./data', target_size=(128, 128), show_stats=True):
    """
    准备数据并创建数据加载器
    """
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        print("请确保已下载数据并放置在正确位置")
        return None, None
    
    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = AnimalDataset(data_dir, transform=train_transform, split='train')
    test_dataset = AnimalDataset(data_dir, transform=test_transform, split='test')
    # 检查数据集是否为空
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("错误: 数据集为空")
        return None, None
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 显示数据统计信息
    if show_stats:
        print(f"\n{'='*50}")
        print("数据统计信息")
        print('='*50)
        print(f"训练集大小: {len(train_dataset)} 张图片")
        print(f"测试集大小: {len(test_dataset)} 张图片")
        print(f"类别数量: {len(ANIMAL_CLASSES)}")
        
        # 统计类别分布
        train_counter = Counter(train_dataset.labels)
        test_counter = Counter(test_dataset.labels)
        print("\n训练集类别分布:")
        for idx, count in sorted(train_counter.items()):
            print(f"  {IDX_TO_CLASS[idx]}: {count} 张 ({count/len(train_dataset)*100:.1f}%)")
        print("\n测试集类别分布:")
        for idx, count in sorted(test_counter.items()):
            print(f"  {IDX_TO_CLASS[idx]}: {count} 张 ({count/len(test_dataset)*100:.1f}%)")
        
        # 可视化数据增强效果
        visualize_data_augmentation(train_dataset)
        # 可视化数据样本
        visualize_data_samples(train_loader)
    
    return train_loader, test_loader

def visualize_data_augmentation(dataset, num_samples=3):
    """可视化数据增强效果"""
    print("\n可视化数据增强效果...")
    
    # 获取原始图像
    original_images = []
    for i in range(num_samples):
        img_path = dataset.images[i]
        original_img = Image.open(img_path).convert('RGB')
        original_images.append(original_img)
    
    # 创建不同的数据增强
    augmentations = {
        '原始图像': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]),
        '水平翻转': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
        ]),
        '旋转': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
        ]),
        '颜色抖动': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
        ]),
        '随机裁剪': transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomCrop((128, 128)),
            transforms.ToTensor(),
        ]),
        '组合增强': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    }
    
    # 创建可视化
    fig, axes = plt.subplots(num_samples, len(augmentations), figsize=(15, 3*num_samples))
    for i, original_img in enumerate(original_images):
        for j, (aug_name, transform) in enumerate(augmentations.items()):
            ax = axes[i, j] if num_samples > 1 else axes[j]
            # 应用增强
            augmented_img = transform(original_img)
            # 反归一化显示
            img_display = augmented_img.numpy().transpose(1, 2, 0)
            img_display = np.clip(img_display, 0, 1)
            
            ax.imshow(img_display)
            if i == 0:  # 只在第一行显示标题
                ax.set_title(aug_name, fontsize=10)
            ax.axis('off')
    
    plt.suptitle('数据增强效果展示', fontsize=14)
    plt.tight_layout()
    plt.savefig('data_augmentation.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_data_samples(dataloader, num_samples=8):
    """可视化数据样本"""
    print("\n可视化数据样本...")
    
    # 获取一个批次的数据
    images, labels = next(iter(dataloader))
    # 反归一化用于显示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flat
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        label_idx = labels[i].item()
        label_name = IDX_TO_CLASS[label_idx]
        
        axes[i].imshow(img)
        axes[i].set_title(f"类别: {label_name}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('训练数据样本展示', fontsize=14)
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载...")
    train_loader, test_loader = prepare_data()
    
    if train_loader and test_loader:
        print("\n数据加载成功！")
        # 显示一个batch的数据
        for images, labels in train_loader:
            print(f"\nBatch信息:")
            print(f"  图像形状: {images.shape}")  # [batch_size, channels, height, width]
            print(f"  标签形状: {labels.shape}")
            print(f"  数据类型: {images.dtype}")
            print(f"  值范围: [{images.min():.3f}, {images.max():.3f}]")
            break
    else:
        print("数据加载失败")