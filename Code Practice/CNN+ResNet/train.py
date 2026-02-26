import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import math
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime

from model import AnimalClassifier
from data_preprocessing import prepare_data, IDX_TO_CLASS

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

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """保存最佳模型"""
        if self.verbose:
            print(f'验证损失下降 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型...')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_loss_min': self.val_loss_min,
            'best_score': self.best_score,
        }, self.path)
        
        self.val_loss_min = val_loss

def train_model(model, train_loader, val_loader, device, num_epochs=20, model_name='cnn_model'):
    """训练函数"""
    # 创建checkpoints目录
    os.makedirs('checkpoints', exist_ok=True)
    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    # 早停机制
    checkpoint_path = f'checkpoints/best_{model_name}.pth'
    early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_path)
    
    # 训练记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': [],
        'epoch_times': []
    }
    # 跟踪最佳模型
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    
    print(f"开始训练 {model_name}...")
    print(f"Epochs: {num_epochs}")
    print(f"Optimizer: AdamW")
    print(f"LR Scheduler: CosineAnnealing")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        # 验证阶段
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        history['epoch_times'].append(time.time() - start_time)
        
        epoch_time = time.time() - start_time
        # 打印进度
        print(f"Epoch {epoch+1:03d}/{num_epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        # 保存最佳模型（基于验证准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            print(f"✓ 新的最佳模型！验证准确率: {val_acc:.2f}% (Epoch {best_epoch})")
        print("-" * 60)
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("早停触发!")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        print(f"加载最佳模型 (Epoch {best_epoch}, 验证准确率: {best_val_acc:.2f}%)")
        model.load_state_dict(best_model_state)
        # 保存最佳模型的完整检查点
        final_checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'val_acc': best_val_acc,
            'val_loss': history['val_loss'][best_epoch-1],
            'train_acc': history['train_acc'][best_epoch-1],
            'train_loss': history['train_loss'][best_epoch-1],
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }
        torch.save(final_checkpoint, checkpoint_path)
        print(f"最佳模型已保存到: {checkpoint_path}")
    else:
        print("警告: 未找到最佳模型，使用最后一个epoch的模型")
        # 保存最后一个epoch的模型
        final_checkpoint = {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        torch.save(final_checkpoint, checkpoint_path)
    
    return history

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # 显示进度条
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            progress = (batch_idx + 1) / len(train_loader)
            print(f"\rEpoch {epoch+1}/{num_epochs} | "
                  f"[{'=' * int(50 * progress):50s}] "
                  f"{progress*100:.1f}% | Loss: {loss.item():.4f}", end='')
    
    print()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, device, class_names, model_type='cnn'):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    print(f"\n{'='*60}")
    print("模型评估结果")
    print('='*60)
    print(f"测试准确率: {accuracy:.2f}%")
    print(f"精确率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1分数: {f1:.2f}%")
    
    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    # 混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, class_names, model_type=model_type)
    # 置信度分析
    plot_confidence_distribution(all_probs, all_labels, all_preds, class_names, model_name=model_type)
    # 保存评估结果
    save_evaluation_results({
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }, class_names)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_confusion_matrix(y_true, y_pred, class_names, model_type='cnn'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title(f'混淆矩阵 - {model_type.upper()}模型性能', fontsize=14, pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_confidence_distribution(probabilities, true_labels, predictions, class_names , model_name='CNN'):
    """绘制置信度分布"""
    # 提取每个样本的最大置信度
    confidences = [np.max(prob) for prob in probabilities]
    # 计算正确和错误预测的置信度
    correct_conf = []
    wrong_conf = []

    for conf, true, pred in zip(confidences, true_labels, predictions):
        if true == pred:
            correct_conf.append(conf)
        else:
            wrong_conf.append(conf)
    
    # 绘制置信度分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 置信度直方图
    axes[0].hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('置信度', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title('所有预测的置信度分布', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    # 正确 vs 错误置信度
    axes[1].hist(correct_conf, bins=20, alpha=0.5, label='正确预测', color='green')
    axes[1].hist(wrong_conf, bins=20, alpha=0.5, label='错误预测', color='red')
    axes[1].set_xlabel('置信度', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].set_title('正确 vs 错误预测的置信度', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'confidence_distribution_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    # 置信度统计
    print("\n置信度统计分析:")
    print(f"  平均置信度: {np.mean(confidences):.3f}")
    print(f"  正确预测平均置信度: {np.mean(correct_conf):.3f}" if correct_conf else "  无正确预测")
    print(f"  错误预测平均置信度: {np.mean(wrong_conf):.3f}" if wrong_conf else "  无错误预测")

def plot_training_history(history, model_name='CNN'):
    """绘制训练历史"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 训练损失
    ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失', fontsize=12)
    ax1.set_title(f'{model_name} - 训练和验证损失曲线', fontsize=14, pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 训练准确率
    ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title(f'{model_name} - 训练和验证准确率曲线', fontsize=14, pad=10)
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率
    ax3.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('学习率', fontsize=12)
    ax3.set_title(f'{model_name} - 学习率变化曲线', fontsize=14, pad=10)
    ax3.grid(True, alpha=0.3)
    
    # 训练时间
    ax4.plot(epochs, history['epoch_times'], 'purple', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('时间 (秒)', fontsize=12)
    ax4.set_title(f'{model_name} - 每个epoch训练时间', fontsize=14, pad=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} 模型训练历史', fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig(f'reports/training_history_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_model(model, model_type, accuracy, path=None):
    """保存模型及元数据"""
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_type == 'cnn_frozen':
            model_type_str = 'resnet_frozen'
        elif model_type == 'cnn':
            model_type_str = 'cnn'
        else:
            model_type_str = model_type
        
        os.makedirs('models', exist_ok=True)
        path = f'models/animal_{model_type_str}_{timestamp}_acc{accuracy:.1f}.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'accuracy': accuracy,
        'num_classes': 5,
        'class_names': ['dog', 'cat', 'horse', 'chicken', 'elephant'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, path)
    
    print(f"模型已保存到 {path}")
    return path

def save_evaluation_results(results, class_names, filename='evaluation_results.json'):
    """保存评估结果"""
    evaluation_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        },
        'class_names': class_names,
        'confusion_matrix': confusion_matrix(results['labels'], results['predictions']).tolist()
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存到 {filename}")

def main():
    """主函数"""
    print("="*60)
    print("CNN && ResNet动物识别系统")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"使用设备: {device}")
    # 准备数据
    print("\n准备数据...")
    train_loader, test_loader = prepare_data()
    if train_loader is None or test_loader is None:
        print("数据加载失败，程序退出")
        return
    
    # 选择模型类型
    print("\n选择模型类型:")
    print("1. cnn - 自定义CNN模型")
    print("2. resnet - ResNet迁移学习模型")
    choice = input("请选择模型类型 (1-2): ").strip()
    model_type = 'resnet' if choice == '2' else 'cnn'
    
    # 如果是ResNet，询问是否冻结backbone
    if model_type == 'resnet':
        freeze_choice = input("是否冻结ResNet的backbone？（冻结适合迁移学习，不冻结适合微调）(y/n): ").strip().lower()
        freeze_backbone = freeze_choice == 'y'
        print(f"\n创建 {model_type} 模型 (冻结backbone: {freeze_backbone})...")
        classifier = AnimalClassifier(
            model_type=model_type, 
            num_classes=5, 
            device=device,
            freeze_backbone=freeze_backbone
        )
    else:
        print(f"\n创建 {model_type} 模型...")
        classifier = AnimalClassifier(
            model_type=model_type, 
            num_classes=5, 
            device=device
        )
    
    # 训练参数配置
    print("\n配置训练参数:")
    try:
        epochs = int(input("输入训练轮数 (默认20): ") or "20")
    except:
        epochs = 20
        print(f"使用默认轮数: {epochs}")
    
    # 训练模型
    print(f"\n训练 {model_type} 模型 (epochs={epochs})...")
    history = train_model(
        classifier.model, train_loader, test_loader, device, 
        num_epochs=epochs, model_name=model_type
    )
    # 绘制训练历史
    plot_training_history(history, model_type.upper())
    
    # 评估模型
    print("\n评估模型性能...")
    eval_results = evaluate_model(classifier.model, test_loader, device, classifier.class_names, model_type=model_type)
    # 可视化预测结果
    print("\n可视化预测结果...")
    classifier.visualize_predictions(test_loader, save_path=f'predictions_{model_type}.png')
    # 保存模型（更新命名）
    model_suffix = "frozen" if (model_type == 'resnet' and freeze_backbone) else model_type
    model_path = save_model(classifier.model, model_suffix, eval_results['accuracy'])
    # 生成训练报告
    generate_training_report(history, eval_results, model_type, model_path)
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

def generate_training_report(history, eval_results, model_type, model_path):
    """生成训练报告"""
    # 转换NumPy类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 准备报告数据
    report = {
        'training_info': {
            'model_type': model_type,
            'final_train_accuracy': float(history['train_acc'][-1]) if history['train_acc'] else 0,
            'final_val_accuracy': float(history['val_acc'][-1]) if history['val_acc'] else 0,
            'total_epochs': int(len(history['train_loss'])),
            'best_epoch': int(np.argmax(history['val_acc']) + 1) if history['val_acc'] else 0,
            'model_path': model_path
        },
        'evaluation_results': {
            'test_accuracy': float(eval_results['accuracy']),
            'precision': float(eval_results['precision']),
            'recall': float(eval_results['recall']),
            'f1_score': float(eval_results['f1_score'])
        },
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']],
            'learning_rate': [float(x) for x in history.get('learning_rate', [])],
            'epoch_times': [float(x) for x in history.get('epoch_times', [])]
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # 应用类型转换
    report = convert_numpy_types(report)
    # 保存报告
    os.makedirs('reports', exist_ok=True)
    with open('training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n训练报告已保存到 training_report.json")
    
    # 打印报告摘要
    print("\n" + "="*60)
    print("训练报告摘要")
    print("="*60)
    print(f"模型类型: {model_type}")
    print(f"训练轮数: {len(history['train_loss'])}")
    print(f"最终训练准确率: {history['train_acc'][-1]:.1f}%")
    print(f"最终验证准确率: {history['val_acc'][-1]:.1f}%")
    print(f"测试准确率: {eval_results['accuracy']:.1f}%")
    print(f"F1分数: {eval_results['f1_score']:.1f}%")
    print(f"模型保存路径: {model_path}")
    print("="*60)

if __name__ == "__main__":
    # 创建所有需要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True) 
    
    main()