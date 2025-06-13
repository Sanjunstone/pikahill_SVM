import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib as mpl
import matplotlib.ticker as mtick
from collections import Counter

# 设置中文字体支持
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# ===================== 配置部分 =====================
DATA_PATHS = {
    "material1": fr"D:\base_TEM\pedict_acc_young_35\dA-training-features.npy",
    "material2": fr"D:\base_TEM\pedict_acc_young_35\dG-training-features.npy",
    "material3": fr"D:\base_TEM\pedict_acc_young_35\dC-training-features.npy",
    "material4": fr"D:\base_TEM\pedict_acc_young_35\dT-training-features.npy",
    "material5": fr"D:\base_TEM\pedict_acc_young_35\dI-training-features.npy",
    "material6": fr"D:\base_TEM\pedict_acc_young_35\dU-training-features.npy",
    "material7": fr"D:\base_TEM\pedict_acc_young_35\8-OHdG-training-features.npy",
    "material8": fr"D:\base_TEM\pedict_acc_young_35\5-hmcdC-training-features.npy",
    "material9": fr"D:\base_TEM\pedict_acc_young_35\m6-dA-training-features.npy",
}

PREDICT_PATH = {
    "material1": fr"D:\base_TEM\pedict_acc_young_35\dA-testing-features.npy",
    "material2": fr"D:\base_TEM\pedict_acc_young_35\dG-testing-features.npy",
    "material3": fr"D:\base_TEM\pedict_acc_young_35\dC-testing-features.npy",
    "material4": fr"D:\base_TEM\pedict_acc_young_35\dT-testing-features.npy",
    "material5": fr"D:\base_TEM\pedict_acc_young_35\dI-testing-features.npy",
    "material6": fr"D:\base_TEM\pedict_acc_young_35\dU-testing-features.npy",
    "material7": fr"D:\base_TEM\pedict_acc_young_35\8-OHdG-testing-features.npy",
    "material8": fr"D:\base_TEM\pedict_acc_young_35\5-hmcdC-testing-features.npy",
    "material9": fr"D:\base_TEM\pedict_acc_young_35\m6-dA-testing-features.npy",
}

CLASS_SETTINGS = {
    "num_classes": 9,
    "class_names": ["dA", "dG", "dC", "dT", "dI", "dU", "8-OHdG", "5-hmcdC","m6-dA"]
}

FEATURE_SELECTION = {
    "method": "mutual_info",
    "n_features": 40,
    "save_path": "selected_features.npy"
}

TORCH_CONFIG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "batch_size": 256,
    "epochs": 500,
    "learning_rate": 0.00001,
    "gamma": 200,
    "rff_dim": 4096,
    "weight_decay": 0.01,
    "model_save_path": "kernel_svm_multiclass.pth",
    "label_smoothing": 0.05
}

CROSS_VAL_SETTINGS = {
    "n_splits": 10,
    "random_state": 42,
    "save_final_model": True
}

# 设置并行处理环境变量
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


# ===================== 数据加载函数 =====================
def load_data(test_size=0.1, random_state=42):
    """使用样本重复处理不平衡数据"""
    # 加载原始完整数据
    materials = [
        np.load(DATA_PATHS[f"material{i}"], allow_pickle=True).astype(np.float32)
        for i in range(1, 10)  # 加载材料
    ]

    # 创建原始数据集
    X_raw = np.vstack(materials)
    y_raw = np.hstack([np.full(len(mat), i - 1) for i, mat in enumerate(materials, 1)])

    # 划分训练集和测试集（保持原始分布）
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw,
        test_size=test_size,
        stratify=y_raw,
        random_state=random_state
    )

    # 标准化处理（使用训练集拟合）
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用样本重复平衡训练集
    print("\n使用样本重复平衡训练集...")
    # 计算每个类别的样本数量
    class_counts = np.bincount(y_train)
    max_count = np.max(class_counts)

    # 存储重复后的数据和标签
    X_resampled = []
    y_resampled = []

    # 对每个类别进行样本重复
    for class_idx in range(len(class_counts)):
        # 获取当前类别的所有样本
        class_indices = np.where(y_train == class_idx)[0]
        X_class = X_train_scaled[class_indices]
        y_class = y_train[class_indices]

        # 计算需要重复的次数
        num_repeats = max_count // len(X_class)
        remainder = max_count % len(X_class)

        # 重复样本
        X_repeated = np.repeat(X_class, num_repeats, axis=0)
        y_repeated = np.repeat(y_class, num_repeats, axis=0)

        # 添加剩余样本
        if remainder > 0:
            indices = np.random.choice(len(X_class), remainder, replace=False)
            X_repeated_add = X_class[indices]
            y_repeated_add = y_class[indices]

            X_repeated = np.vstack([X_repeated, X_repeated_add])
            y_repeated = np.hstack([y_repeated, y_repeated_add])

        X_resampled.append(X_repeated)
        y_resampled.append(y_repeated)

    # 合并所有类别的数据
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.hstack(y_resampled)

    # 打乱数据顺序
    indices = np.arange(len(X_resampled))
    np.random.shuffle(indices)
    X_resampled = X_resampled[indices]
    y_resampled = y_resampled[indices]

    # 打印类别分布变化
    orig_counts = np.bincount(y_train)
    new_counts = np.bincount(y_resampled)
    print(f"原始类别分布: {orig_counts}")
    print(f"平衡后类别分布: {new_counts}")

    return X_resampled, X_test_scaled, y_resampled, y_test


def load_predict_data():
    """加载预测数据"""
    # 加载8个材料数据
    materials = [
        np.load(PREDICT_PATH[f"material{i}"], allow_pickle=True).astype(np.float32)
        for i in range(1, 10)
    ]

    # 合并数据
    X = np.vstack(materials)
    y = np.hstack([np.full(len(mat), i - 1) for i, mat in enumerate(materials, 1)])

    print(f"预测数据加载完成，形状: {X.shape}")
    return X, y


# ===================== 特征选择函数 =====================
def select_features(X_train, y_train, method='mutual_info', n_features=30):
    """多分类特征选择"""
    assert X_train.shape[1] >= n_features, "特征数量超过数据维度"

    if method == 'mutual_info':
        mi_scores = mutual_info_classif(X_train, y_train)
        return np.argsort(mi_scores)[-n_features:]

    elif method == 'l1_svm':
        selector = SelectFromModel(
            LinearSVC(C=0.01, penalty='l1', dual=False, multi_class='ovr'),
            max_features=n_features
        )
        selector.fit(X_train, y_train)
        return selector.get_support(indices=True)

    elif method == 'random_forest':
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100),
            max_features=n_features
        )
        selector.fit(X_train, y_train)
        return selector.get_support(indices=True)

    else:
        raise ValueError("不支持的筛选方法")


# ===================== PyTorch模型 =====================
class KernelSVM(nn.Module):
    """多分类核SVM"""

    def __init__(self, input_dim):
        super().__init__()
        # RFF参数初始化
        self.register_buffer("omega",
                             torch.randn(input_dim, TORCH_CONFIG["rff_dim"]) /
                             (2 * TORCH_CONFIG["gamma"]) ** 0.5)
        self.register_buffer("b",
                             torch.rand(TORCH_CONFIG["rff_dim"]) * 2 * torch.pi)

        # 分类网络
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(TORCH_CONFIG["rff_dim"]),
            nn.Linear(TORCH_CONFIG["rff_dim"], 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, CLASS_SETTINGS["num_classes"])
        )

    def _rff_transform(self, x):
        """特征变换"""
        proj = x @ self.omega + self.b
        return torch.cos(proj) * (2 / TORCH_CONFIG["rff_dim"]) ** 0.5

    def forward(self, x):
        features = self._rff_transform(x)
        return self.classifier(features)


def train_model(X_train, y_train, selected_features):
    """训练函数"""
    # 数据准备
    X = X_train[:, selected_features].astype(np.float32)
    X_tensor = torch.FloatTensor(X).to(TORCH_CONFIG["device"])
    y_tensor = torch.LongTensor(y_train).to(TORCH_CONFIG["device"])

    # 初始化组件
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=TORCH_CONFIG["batch_size"],
        shuffle=True,
        pin_memory=False
    )

    model = KernelSVM(X.shape[1]).to(TORCH_CONFIG["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TORCH_CONFIG["learning_rate"],
        weight_decay=TORCH_CONFIG["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=TORCH_CONFIG["label_smoothing"])

    # 训练循环
    best_acc = 0.0
    training_log = []
    for epoch in range(TORCH_CONFIG["epochs"]):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()

        # 每10个epoch验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                _, preds = torch.max(outputs, 1)
                acc = (preds == y_tensor).float().mean()
                avg_loss = total_loss / len(loader)

            training_log.append([epoch + 1, avg_loss, acc.cpu().numpy()])
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), TORCH_CONFIG["model_save_path"])

    np.save("train_data.npy", np.array(training_log))
    return model


# ===================== 模型评估函数 =====================
def evaluate_model(model, X_test, y_test, selected_features):
    """多分类评估"""
    model.eval()

    # 数据预处理
    X = X_test[:, selected_features].astype(np.float32)
    X_tensor = torch.FloatTensor(X).to(TORCH_CONFIG["device"])

    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, preds.cpu().numpy(),
                                target_names=CLASS_SETTINGS["class_names"]))

    # 混淆矩阵
    cm = confusion_matrix(y_test, preds.cpu().numpy())
    # 将混淆矩阵元素转换为百分比
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.where(row_sums > 0, cm.astype('float') / row_sums, 0)

    # 构建注释文本
    torch_config_text = "\n".join([f"TORCH_CONFIG - {key}: {value}" for key, value in TORCH_CONFIG.items()])
    feature_selection_text = "\n".join(
        [f"FEATURE_SELECTION - {key}: {value}" for key, value in FEATURE_SELECTION.items()])
    config_text = torch_config_text + "\n\n" + feature_selection_text

    # 调整图表大小，为左侧注释框留出空间
    plt.figure(figsize=(18, 9))
    plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("混淆矩阵", fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_SETTINGS["class_names"]))
    plt.xticks(tick_marks, CLASS_SETTINGS["class_names"], rotation=45, fontsize=14)
    plt.yticks(tick_marks, CLASS_SETTINGS["class_names"], fontsize=14)

    # 添加配置文本
    plt.annotate(config_text, xy=(-1, 0.8), xycoords='axes fraction',
                 bbox=dict(boxstyle='round,pad=0.5', edgecolor='gray', facecolor='white', alpha=1),
                 va='center', ha='left', fontsize=10)

    # 调整子图布局
    plt.subplots_adjust(left=0.4)

    # 添加百分比注释
    thresh = cm_percentage.max() / 2.
    for i in range(cm_percentage.shape[0]):
        for j in range(cm_percentage.shape[1]):
            plt.text(j, i, format(cm_percentage[i, j] * 100, '.2f') + '%',
                     horizontalalignment="center",
                     color="white" if cm_percentage[i, j] > thresh else "black",
                     fontsize=14)

    plt.ylabel('真实标签', fontsize=14)
    plt.xlabel('预测标签', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return cm


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    print(f"数据加载完成，训练集形状: {X_train.shape}，测试集形状: {X_test.shape}")
    print(f"类别分布 - 训练集: {np.bincount(y_train)}，测试集: {np.bincount(y_test)}")

    # 特征筛选
    selected_features = select_features(
        X_train, y_train,
        method=FEATURE_SELECTION["method"],
        n_features=FEATURE_SELECTION["n_features"]
    )
    np.save(FEATURE_SELECTION["save_path"], selected_features)
    print(f"\n筛选出{len(selected_features)}个特征，索引: {selected_features}")

    # 训练模型
    print("\n开始训练...")
    model = train_model(X_train, y_train, selected_features)

    # 评估模型
    print("\n模型评估中...")
    cm = evaluate_model(model, X_test, y_test, selected_features)

    # 加载独立测试集进行评估
    print("\n加载独立测试集进行评估...")
    X_independent, y_independent = load_predict_data()
    cm_independent = evaluate_model(model, X_independent, y_independent, selected_features)