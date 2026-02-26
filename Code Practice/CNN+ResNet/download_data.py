# download_data.py
# 该脚本用于下载并整理Animals-10数据集

import os
import zipfile
import kaggle
from pathlib import Path
import shutil
import random

def setup_kaggle():
    """设置Kaggle API"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # 检查kaggle.json是否存在
    if not (kaggle_dir / 'kaggle.json').exists():
        print("请先设置Kaggle API Token:")
        print("1. 访问 https://www.kaggle.com/account")
        print("2. 创建API Token")
        print("3. 下载kaggle.json")
        print("4. 复制到 ~/.kaggle/")
        return False
    return True

def download_dataset():
    """下载Animals-10数据集"""
    print("正在下载Animals-10数据集...")
    
    try:
        # 使用kaggle API下载
        kaggle.api.dataset_download_files('alessiocorrado99/animals10', path='./', unzip=True)
        print("下载完成！")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

def reorganize_for_simple_structure():
    """重新组织数据集为简单结构"""
    print("\n重新组织数据集结构...")
    # 原始意大利语到英语的映射
    italian_to_english = {
        'cane': 'dog',
        'gatto': 'cat',
        'cavallo': 'horse',
        'elefante': 'elephant',
        'farfalla': 'butterfly',
        'gallina': 'chicken',
        'mucca': 'cow',
        'pecora': 'sheep',
        'ragno': 'spider',
        'scoiattolo': 'squirrel'
    }
    
    # 只选择5种常见动物
    selected_classes = ['dog', 'cat', 'horse', 'chicken', 'elephant']
    # 创建简单结构目录
    simple_data_dir = Path('./data')
    simple_data_dir.mkdir(exist_ok=True)
    # 为每个选定类别创建目录
    for english_name in selected_classes:
        class_dir = simple_data_dir / english_name
        class_dir.mkdir(exist_ok=True)
    total_images = 0
    category_stats = {}
    
    # 遍历原始数据目录
    source_dir = Path('./raw-img')
    if not source_dir.exists():
        print(f"错误：原始数据目录 '{source_dir}' 不存在")
        return False
    print(f"从 {source_dir} 处理图像...")
    
    # 处理每个意大利语类别
    for italian_name, english_name in italian_to_english.items():
        # 只处理选定的类别
        if english_name not in selected_classes:
            continue
        source_class_dir = source_dir / italian_name
        if not source_class_dir.exists():
            print(f"警告：目录 '{source_class_dir}' 不存在，跳过")
            continue
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(source_class_dir.glob(ext)))
        if not image_files:
            print(f"警告：类别 {english_name} ({italian_name}) 中没有找到图像")
            continue
        
        # 随机打乱图像
        random.shuffle(image_files)
        # 复制图像到目标目录
        target_class_dir = simple_data_dir / english_name
        copied_count = 0
        for img_path in image_files:
            try:
                # 保持原文件名或生成新文件名
                filename = f"{english_name}_{copied_count:04d}{img_path.suffix}"
                target_path = target_class_dir / filename
                # 复制文件
                shutil.copy2(img_path, target_path)
                copied_count += 1
            except Exception as e:
                print(f"复制文件错误 {img_path}: {e}")
        
        total_images += copied_count
        category_stats[english_name] = copied_count
        print(f"类别 {english_name}: 复制了 {copied_count} 张图片")
    
    # 保存数据集信息
    dataset_info = {
        'total_images': total_images,
        'categories': selected_classes,
        'category_counts': category_stats,
        'source': 'Animals-10 from Kaggle',
        'processed_date': '自动处理'
    }
    # 保存为JSON文件
    info_file = simple_data_dir / 'dataset_info.json'
    import json
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
    print(f"\n数据集组织完成！")
    print(f"总共 {total_images} 张图片")
    print(f"保存到: {simple_data_dir}")
    
    # 显示目录结构
    print("\n生成的目录结构:")
    for class_name in selected_classes:
        class_dir = simple_data_dir / class_name
        count = len(list(class_dir.glob('*.*')))
        print(f"  data/{class_name}/ - {count} 张图片")
    print(f"\n数据集信息保存到: {info_file}")
    
    return True

def create_sample_split(train_ratio=0.8):
    """创建训练/测试分割"""
    print("\n创建训练测试分割（可选步骤）...")
    
    data_dir = Path('./data')
    # 创建分割目录
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    total_train = 0
    total_test = 0
    
    # 对每个类别进行分割
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir() and class_dir.name not in ['train', 'test']:
            class_name = class_dir.name
            # 获取所有图像
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            random.shuffle(images)
            # 分割点
            split_idx = int(len(images) * train_ratio)
            # 创建类别目录
            train_class_dir = train_dir / class_name
            test_class_dir = test_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)
            
            # 复制训练集
            for img_path in images[:split_idx]:
                shutil.copy2(img_path, train_class_dir / img_path.name)
                total_train += 1
            # 复制测试集
            for img_path in images[split_idx:]:
                shutil.copy2(img_path, test_class_dir / img_path.name)
                total_test += 1
            print(f"类别 {class_name}: {len(images[:split_idx])} 训练, {len(images[split_idx:])} 测试")
    
    if total_train > 0 and total_test > 0:
        print(f"\n分割完成: {total_train} 训练图像, {total_test} 测试图像")
    return total_train, total_test

def cleanup_intermediate_files():
    """清理中间文件"""
    print("\n清理中间文件...")
    
    # 删除原始解压目录
    raw_dir = Path('./raw-img')
    if raw_dir.exists():
        print(f"删除原始目录: {raw_dir}")
        shutil.rmtree(raw_dir)
    # 删除可能的zip文件
    zip_file = Path('./animals10.zip')
    if zip_file.exists():
        print(f"删除zip文件: {zip_file}")
        os.remove(zip_file)
    print("清理完成！")

def main():
    """主函数"""
    print("="*60)
    print("Animals-10 数据集下载和准备工具")
    print("="*60)
    
    # 步骤1: 设置Kaggle API
    print("\n[1/4] 检查Kaggle API设置...")
    if not setup_kaggle():
        print("Kaggle API设置失败，请手动下载数据集")
        print("从 https://www.kaggle.com/alessiocorrado99/animals10 下载")
        print("解压后将 'raw-img' 目录放在当前文件夹")
        return
    # 步骤2: 下载数据集
    print("\n[2/4] 下载数据集...")
    if not download_dataset():
        print("下载失败，请检查网络连接和API设置")
        return
    # 步骤3: 重新组织数据结构
    print("\n[3/4] 组织数据集结构...")
    if not reorganize_for_simple_structure():
        print("数据组织失败")
        return
    # 步骤4: 可选创建预分割
    print("\n[4/4] 可选步骤: 创建预分割...")
    create_split = input("是否创建预分割的训练/测试目录? (y/n): ").lower().strip()
    if create_split == 'y':
        create_sample_split()
    # 清理中间文件
    cleanup = input("\n是否清理原始数据文件? (y/n): ").lower().strip()
    if cleanup == 'y':
        cleanup_intermediate_files()
    
    print("\n" + "="*60)
    print("数据集准备完成!")
    print("="*60)
    print("\n您现在可以运行第一份代码:")
    print("1. 确保有以下目录结构:")
    print("   data/")
    print("   ├── dog/")
    print("   ├── cat/")
    print("   ├── horse/")
    print("   ├── chicken/")
    print("   └── elephant/")
    print("\n2. 运行以下命令:")
    print("   python data_preprocessing.py  # 测试数据加载")
    print("   python train_cnn.py           # 训练CNN模型")
    print("="*60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()