import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import random
import matplotlib

# 设置中文字体
def setup_chinese_font():
    try:
        # 检查系统中是否有中文字体
        available_fonts = []
        # 常见的中文字体名称
        chinese_fonts = [
            'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 
            'STHeiti', 'AR PL UMing CN', 'AR PL UKai CN', 'DejaVu Sans' 
        ]
        # 检查系统中哪些字体可用
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        available_fonts = [f.name for f in fm.ttflist]
        # 选择第一个可用的中文字体
        selected_font = 'DejaVu Sans'  # 默认
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        print(f"使用字体: {selected_font}")
        
        # 设置matplotlib字体
        plt.rcParams['font.sans-serif'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        # 设置matplotlib使用的字体
        matplotlib.rcParams['font.sans-serif'] = [selected_font]
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        return True
    except Exception as e:
        print(f"字体设置失败: {e}")
        print("将使用默认英文字体")
        return False

setup_chinese_font()

def load_model(model_path):
    """加载训练好的模型"""
    # 自动检测可用设备
    if torch.cuda.is_available():
        device = 'cuda'
        map_location = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        map_location = 'mps'
    else:
        device = 'cpu'
        map_location = 'cpu'
    print(f"检测到设备: {device}")
    print(f"加载模型: {os.path.basename(model_path)}")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=map_location)
    # 获取模型信息
    model_type = checkpoint.get('model_type', 'cnn')
    num_classes = checkpoint.get('num_classes', 5)
    print(f"模型类型: {model_type}")
    print(f"类别数量: {num_classes}")
    print(f"训练准确率: {checkpoint.get('accuracy', '未知'):.1f}%")
    print(f"训练时间: {checkpoint.get('timestamp', '未知')}")
    
    # 导入模型类
    try:
        from model import AnimalClassifier
        classifier = AnimalClassifier(
            model_type=model_type,
            num_classes=num_classes,
            device=device
        )
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.model.eval()
        return classifier
    except ImportError as e:
        print(f"无法导入模型类: {e}")
        print("请确保 cnn_model.py 文件存在且包含 AnimalClassifier 类")
        return None

def test_single_image(classifier, image_path):
    """测试单张图像"""
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在: {image_path}")
        return None
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        # 加载和预处理图像
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
        # 使用检测到的设备
        device = next(classifier.model.parameters()).device
        img_tensor = img_tensor.to(device)
        # 预测
        with torch.no_grad():
            outputs = classifier.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        # 类别名称（如果模型有class_names属性就使用，否则使用默认）
        if hasattr(classifier, 'class_names'):
            class_names = classifier.class_names
        else:
            class_names = ['dog', 'cat', 'horse', 'chicken', 'elephant']
        # 获取Top-3预测
        top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
        top3_prob = top3_prob[0].cpu().numpy()
        top3_idx = top3_idx[0].cpu().numpy()
        
        # 整理结果
        result = {
            'class_name': class_names[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': probabilities[0].cpu().numpy(),
            'top3': [(class_names[idx], prob) for idx, prob in zip(top3_idx, top3_prob)]
        }
        # 显示结果
        print(f"\n{'='*50}")
        print(f"图像: {os.path.basename(image_path)}")
        print(f"预测结果: {result['class_name']}")
        print(f"置信度: {result['confidence']:.2%}")
        print("\nTop-3预测:")
        for i, (cls, prob) in enumerate(result['top3'], 1):
            print(f"  {i}. {cls}: {prob:.2%}")
        print('='*50)
        # 可视化结果
        visualize_prediction(img, result)
        return result
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None

def visualize_prediction(image, result):
    """可视化预测结果"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # 显示图像
        axes[0].imshow(image)
        axes[0].set_title(f'输入图像', fontsize=12)
        axes[0].axis('off')
        # 显示预测结果条形图
        class_names = ['dog', 'cat', 'horse', 'chicken', 'elephant']
        probs = result['probabilities']
        y_pos = np.arange(len(class_names))
        bars = axes[1].barh(y_pos, probs, color='skyblue')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(class_names)
        axes[1].set_xlabel('概率', fontsize=12)
        axes[1].set_title(f'预测概率分布', fontsize=12)
        axes[1].set_xlim([0, 1])
        # 高亮最高概率的类别
        max_idx = np.argmax(probs)
        bars[max_idx].set_color('red')
        # 添加概率标签
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{prob:.3f}', va='center', fontsize=10,
                        color='red' if i == max_idx else 'black')
        
        plt.suptitle('动物识别模型预测结果', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"可视化失败: {e}")

def test_batch_images(classifier, data_dir, num_images=10):
    """批量测试图像"""
    print(f"\n开始批量测试 {num_images} 张图像...")
    print(f"数据目录: {data_dir}")
    # 收集所有图像文件
    image_files = []
    # 检查数据目录结构
    if os.path.exists(data_dir):
        # 如果目录下有子目录（按类别）
        subdirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        if subdirs:  # 有子目录结构
            print(f"找到 {len(subdirs)} 个子目录")
            for subdir in subdirs:
                subdir_path = os.path.join(data_dir, subdir)
                images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                image_files.extend(images)
        else:  # 直接是图片文件
            images = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            image_files.extend(images)
    
    if not image_files:
        print(f"错误: 在 {data_dir} 中没有找到图像文件")
        return []
    
    print(f"总共找到 {len(image_files)} 张图像")
    
    # 随机选择图像
    if len(image_files) < num_images:
        print(f"警告: 只有 {len(image_files)} 张图像可用，将测试所有图像")
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, num_images)
    # 测试每张图像
    results = []
    for i, img_path in enumerate(selected_images, 1):
        print(f"\n测试进度: {i}/{len(selected_images)}")
        
        # 从路径推断真实标签（如果有子目录）
        true_label = None
        if '\\' in img_path or '/' in img_path:
            # 尝试从路径中提取类别
            parts = img_path.replace('\\', '/').split('/')
            for part in parts:
                if part in ['dog', 'cat', 'horse', 'chicken', 'elephant']:
                    true_label = part
                    break
        
        # 测试图像
        result = test_single_image(classifier, img_path)
        if result:
            is_correct = (true_label == result['class_name']) if true_label else None
            results.append({
                'image': os.path.basename(img_path),
                'path': img_path,
                'true_label': true_label,
                'prediction': result['class_name'],
                'confidence': result['confidence'],
                'correct': is_correct
            })
    
    # 打印批量测试总结
    if results:
        print(f"\n{'='*60}")
        print(f"批量测试完成！共测试 {len(results)} 张图像")
        print('='*60)
        # 计算统计数据
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"平均置信度: {avg_confidence:.2%}")
        # 如果有真实标签，计算准确率
        labeled_results = [r for r in results if r['true_label'] is not None]
        if labeled_results:
            correct = sum(1 for r in labeled_results if r['correct'] is True)
            accuracy = correct / len(labeled_results) * 100
            print(f"准确率: {accuracy:.1f}% ({correct}/{len(labeled_results)})")
        # 保存结果
        save_results(results)
    
    return results

def save_results(results):
    """保存测试结果"""
    os.makedirs('test_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'test_results/test_results_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总图像数: {len(results)}\n\n")
        for r in results:
            status = "✓" if r.get('correct') is True else "✗" if r.get('correct') is False else "?"
            f.write(f"{status} {r['image']}\n")
            f.write(f"  预测: {r['prediction']} ({r['confidence']:.2%})\n")
            if r['true_label']:
                f.write(f"  真实: {r['true_label']}\n")
            f.write(f"  路径: {r['path']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\n详细结果已保存到: {filename}")

def main():
    """主函数"""
    print("="*60)
    print("动物识别模型测试系统")
    print("="*60)
    
    # 1. 查找模型文件
    model_files = []
    search_dirs = ['models', '.']
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for f in os.listdir(search_dir):
                if f.endswith('.pth') and ('animal_' in f or 'model_' in f):
                    full_path = os.path.join(search_dir, f)
                    model_files.append(full_path)
    
    if not model_files:
        print("错误: 没有找到模型文件 (.pth)")
        print("请确保:")
        print("1. 模型文件在 models/ 目录或当前目录")
        print("2. 文件名包含 'animal_' 或 'model_'")
        return
    
    # 2. 选择模型
    print("\n找到的模型文件:")
    for i, f in enumerate(model_files[:5]):  # 只显示前5个
        filename = os.path.basename(f)
        print(f"{i+1}. {filename}")
    if len(model_files) > 5:
        print(f"... 还有 {len(model_files)-5} 个模型")
        
    choice = input("\n请选择模型文件 (输入序号，默认1): ").strip()
    try:
        selected_idx = int(choice) - 1 if choice else 0
        selected_model = model_files[selected_idx]
    except:
        selected_model = model_files[0]
    
    # 3. 加载模型
    classifier = load_model(selected_model)
    if classifier is None:
        return
    
    # 4. 选择测试模式
    print("\n选择测试模式:")
    print("1. 测试单张图像")
    print("2. 批量测试")
    mode = input("请选择模式 (1-2，默认1): ").strip()
    if mode == '2':
        data_dir = input("请输入测试数据目录 (默认 'data'): ").strip()
        if not data_dir:
            data_dir = 'data'
        if not os.path.exists(data_dir):
            print(f"错误: 目录不存在: {data_dir}")
            print("请将测试图片放在 data/ 目录下")
            return
        
        num_images = input("请输入测试图像数量 (默认5): ").strip()
        try:
            num_images = int(num_images) if num_images else 5
        except:
            num_images = 5
        
        test_batch_images(classifier, data_dir, num_images)
    
    else:
        # 测试单张图像
        print("\n单张图像测试")
        print("提示: 你可以直接输入:")
        print("  - 完整路径，如: data/test/dog/dog_001.jpg")
        print("  - 相对路径，如: data/dog_001.jpg")
        print("  - 或直接在 data/ 目录下选择图片")
        
        # 显示 data/ 目录下的图片
        if os.path.exists('data'):
            print("\ndata/ 目录下的图片:")
            image_files = []
            for root, dirs, files in os.walk('data'):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rel_path = os.path.relpath(os.path.join(root, file), '.')
                        image_files.append(rel_path)

            if image_files:
                for i, img in enumerate(image_files[:10]):  # 显示前10个
                    print(f"{i+1}. {img}")
                if len(image_files) > 10:
                    print(f"... 还有 {len(image_files)-10} 个图片")
        
        while True:
            image_path = input("\n请输入图像路径 (或输入 'quit' 退出): ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
            
            # 如果输入的是数字，选择显示的图片
            if image_path.isdigit():
                idx = int(image_path) - 1
                if 0 <= idx < len(image_files):
                    image_path = image_files[idx]
            if os.path.exists(image_path):
                test_single_image(classifier, image_path)
            else:
                print(f"错误: 图像不存在: {image_path}")
                print("请检查路径是否正确")

if __name__ == "__main__":
    main()