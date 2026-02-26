import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class CNN(nn.Module):
    """CNN模型"""
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(CNN, self).__init__()
        
        # 卷积块1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate/2)
        )
        
        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate)
        )
        
        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(dropout_rate)
        )
        
        # 卷积块4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(dropout_rate)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 卷积特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接分类
        x = self.fc_layers(x)
        
        return x
    
    def get_feature_maps(self, x):
        """获取各层的特征图"""
        features = []
        # 获取每个卷积块的输出
        x1 = self.conv_block1(x)
        features.append(x1)
        x2 = self.conv_block2(x1)
        features.append(x2)
        x3 = self.conv_block3(x2)
        features.append(x3)
        x4 = self.conv_block4(x3)
        features.append(x4)
        
        return features

class ResNet(nn.Module):
    """基于预训练ResNet的模型（迁移学习）"""
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=False):
        super(ResNet, self).__init__()
        
        try:
            if pretrained:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.resnet = models.resnet18(weights=None)
        except (AttributeError, TypeError):
            self.resnet = models.resnet18(pretrained=pretrained)
        
        # 冻结前面层的参数
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # 替换最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class AnimalClassifier:
    """动物分类器封装类"""
    def __init__(self, model_type='cnn', num_classes=5, device=None, **kwargs):
        self.num_classes = num_classes
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 选择模型类型
        if model_type == 'cnn':
            self.model = CNN(num_classes, **kwargs).to(self.device)
        elif model_type == 'resnet':
            # 获取是否冻结backbone的参数
            freeze_backbone = kwargs.get('freeze_backbone', False)
            self.model = ResNet(num_classes, freeze_backbone=freeze_backbone).to(self.device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，可选 'cnn' 或 'resnet'")
        
        self.class_names = ['dog', 'cat', 'horse', 'chicken', 'elephant']
        self.model_type = model_type
        
        print(f"使用设备: {self.device}")
        print(f"模型类型: {model_type}")
        print(f"参数数量: {self._count_parameters():,}")
        
        # 打印模型结构
        self.print_model_summary()
    
    def _count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def print_model_summary(self):
        """打印模型结构摘要"""
        print("\n模型结构摘要:")
        print("-" * 50)
        
        try:
            summary(self.model, input_size=(3, 128, 128), device=self.device)
        except:
            print(f"输入: (batch_size, 3, 128, 128)")
            
            if isinstance(self.model, CNN):
                print("卷积块1输出: (batch_size, 32, 64, 64)")
                print("卷积块2输出: (batch_size, 64, 32, 32)")
                print("卷积块3输出: (batch_size, 128, 16, 16)")
                print("卷积块4输出: (batch_size, 256, 4, 4)")
                print(f"全连接层输出: (batch_size, {self.num_classes})")
            elif isinstance(self.model, ResNet):
                print("ResNet18 backbone")
                print(f"全连接层输出: (batch_size, {self.num_classes})")
            
            print(f"总参数量: {self._count_parameters():,}")
            print(f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        print("-" * 50)
    
    def predict(self, image_tensor):
        """预测单张图像"""
        self.model.eval()
        with torch.no_grad():
            # 确保有batch维度
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # 获取top-3预测
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            
        return {
            'class_idx': predicted.item(),
            'class_name': self.class_names[predicted.item()],
            'confidence': probabilities.max().item(),
            'probabilities': probabilities.cpu().numpy()[0],
            'top3': [
                (self.class_names[idx], prob.item())
                for idx, prob in zip(top3_indices[0], top3_probs[0])
            ]
        }
    
    def predict_batch(self, batch_tensor):
        """预测批量图像"""
        self.model.eval()
        with torch.no_grad():
            batch_tensor = batch_tensor.to(self.device)
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
        return {
            'predictions': predicted.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'confidences': probabilities.max(dim=1).values.cpu().numpy()
        }
    
    def visualize_predictions(self, dataloader, num_samples=8, save_path='predictions.png'):
        """可视化预测结果"""
        import matplotlib.pyplot as plt
        
        self.model.eval()
        images, true_labels = next(iter(dataloader))
        
        # 预测
        results = self.predict_batch(images[:num_samples])
        # 反归一化用于显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        display_images = images[:num_samples] * std + mean
        display_images = display_images.clamp(0, 1)
        # 创建可视化
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.flat
        
        for i in range(num_samples):
            img = display_images[i].permute(1, 2, 0).numpy()
            true_label = self.class_names[true_labels[i].item()]
            pred_label = self.class_names[results['predictions'][i]]
            confidence = results['confidences'][i]
            
            # 设置颜色（绿色正确，红色错误）
            color = 'green' if true_label == pred_label else 'red'
            axes[i].imshow(img)
            axes[i].set_title(
                f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.2%}',
                color=color, fontsize=10
            )
            axes[i].axis('off')
        
        plt.suptitle(f'{self.model_type.upper()}模型预测结果可视化', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # 计算准确率
        correct = (results['predictions'] == true_labels[:num_samples].numpy()).sum()
        accuracy = correct / num_samples * 100
        print(f"样本准确率: {accuracy:.1f}% ({correct}/{num_samples})")

if __name__ == "__main__":
    # 测试所有模型
    print("测试CNN和ResNet模型...")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    # 创建测试输入，并放在正确的设备上
    test_input = torch.randn(4, 3, 128, 128).to(device)
    
    # 测试各种模型
    model_types = ['cnn', 'resnet']
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"测试模型: {model_type}")
        
        if model_type == 'resnet':
            # 测试带冻结和不冻结的ResNet
            for freeze in [True, False]:
                print(f"\n{'='*40}")
                print(f"ResNet冻结backbone: {freeze}")
                classifier = AnimalClassifier(
                    model_type=model_type, 
                    num_classes=5, 
                    device=device,
                    freeze_backbone=freeze
                )
                output = classifier.model(test_input)
                print(f"输入形状: {test_input.shape}")
                print(f"输出形状: {output.shape}")
                print(f"设备: {test_input.device} -> {output.device}")
        else:
            classifier = AnimalClassifier(
                model_type=model_type, 
                num_classes=5, 
                device=device
            )
            output = classifier.model(test_input)
            print(f"输入形状: {test_input.shape}")
            print(f"输出形状: {output.shape}")
            print(f"设备: {test_input.device} -> {output.device}")