"""
Week 4 (Lab 4): 使用 Keras/TensorFlow 训练简单的 2D-CNN 识别器（孤立词）
- 读取 MFCC 特征 (.npy)
- 填充到统一帧长
- One-Hot 编码
- 划分 train/val/test = 80/10/10
- 构建/训练/评估 CNN，并保存权重
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from utils import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# 为了兼容更多环境，统一使用 tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from tensorflow.keras.optimizers import Adam


# ========= 辅助：确定 MFCC 目录（优先文档的 ./mfccs，其次兼容你 config.MFCC_DIR） =========
def resolve_mfcc_dir() -> Path:
    doc_dir = config.PROJECT_ROOT / "mfccs"  # 文档默认目录
    if doc_dir.exists():
        return doc_dir
    return Path(getattr(config, "MFCC_DIR", config.PROJECT_ROOT / "features" / "mfccs"))


# ========= 核心：加载与填充 =========
def load_mfcc_dir(mfcc_dir: Path) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, int]:
    """
    读取目录下所有 .npy 特征，标签从文件名 stem 的第一个下划线字段解析（期望为 '0'..'9'）。
    - 自动识别输入朝向：
      * 若数据为 [T, D]（如 Week3 保存的默认），自动转为 [D, T]；
      * 若数据已是 [D, T]（文档默认 D=40），则保持不变。
    - 统一填充到相同的帧长 max_frames（沿时间维）。
    - 返回:
        data_4d: (N, 40, max_frames, 1)
        labels_1hot: (N, 10)
        LE: 已拟合的 LabelEncoder（classes = ['0', ... , '9']）
        max_frames: 统一后的帧长
    """
    files = sorted(Path(mfcc_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"没有在 {mfcc_dir} 找到 .npy 特征文件。")

    # 先扫 max_frames
    max_frames = 0
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2:
            raise ValueError(f"特征矩阵必须是 2D：{f} 得到 {arr.ndim}D。")
        if arr.shape[1] == 40:      # [T, D]
            frames = arr.shape[0]
        elif arr.shape[0] == 40:    # [D, T]
            frames = arr.shape[1]
        else:
            raise ValueError(f"MFCC 特征不是 40 维：{f} 形状 {arr.shape}。请用 40 维 MFCC。")
        if frames > max_frames:
            max_frames = frames

    print(f"[Week4] Max frames across dataset = {max_frames}")

    # 再读 + pad + 组装
    data_list: List[np.ndarray] = []
    labels_list: List[str] = []
    for f in files:
        x = np.load(f)  # 2D
        # 统一为 [D(40), T]
        if x.shape[1] == 40:       # [T, D] -> 转为 [D, T]
            x = x.T
        elif x.shape[0] == 40:     # 已是 [D, T]
            pass
        else:
            raise ValueError(f"MFCC 特征不是 40 维：{f} 形状 {x.shape}。")

        D, T = x.shape
        if T < max_frames:
            x = np.pad(x, ((0, 0), (0, max_frames - T)), mode="constant")
        data_list.append(x.astype(np.float32))

        # 标签来自文件名的第一个字段，如 "3_s1_0001.npy" -> '3'
        stem = f.stem
        parts = stem.split("_")
        if len(parts) == 0:
            raise ValueError(f"无法从文件名解析标签：{f}")
        labels_list.append(parts[0])

    # 全数据归一化到 [0,1]
    data = np.stack(data_list, axis=0)  # [N, 40, max_frames]
    global_max = float(np.max(data)) or 1.0
    data = data / global_max
    data = np.expand_dims(data, axis=-1)  # -> [N, 40, max_frames, 1]

    # One-Hot（固定十类 0..9；若你只做子集，可改成按实际类拟合）
    LE = LabelEncoder().fit([str(i) for i in range(10)])
    labels_1hot = to_categorical(LE.transform(labels_list), num_classes=10)

    return data.astype(np.float32), labels_1hot.astype(np.float32), LE, max_frames


# ========= 模型 =========
def create_model(input_shape, num_classes: int = 10) -> Sequential:
    """Conv2D(64,3x3)->MaxPool(3x3)->Flatten->Dense(256)->ReLU->Dense(C)->Softmax"""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(256), Activation('relu'),
        Dense(num_classes), Activation('softmax')
    ])
    return model


# ========= 训练与评估 =========
def train_eval(mfcc_dir: Path | None = None, plot_curves: bool = True):
    # 目录解析（与文档/前几周都兼容）
    mfcc_dir = Path(mfcc_dir) if mfcc_dir else resolve_mfcc_dir()
    print(f"[Week4] Using MFCC directory: {mfcc_dir}")

    # 加载数据
    X, y, LE, max_frames = load_mfcc_dir(mfcc_dir)
    print(f"[Week4] Data shape (N, 40, {max_frames}, 1) = {X.shape}, Labels = {y.shape}")

    # 80/10/10 切分（文档做法：两次 train_test_split）
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=config.RANDOM_STATE)

    # 构建与编译
    lr = getattr(config, "LEARNING_RATE", 1e-2)
    batch = getattr(config, "BATCH_SIZE", 32)
    epochs = getattr(config, "NUM_EPOCHS", 25)

    model = create_model(input_shape=X_train.shape[1:], num_classes=y.shape[1])
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=lr))
    model.summary()

    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch,
        epochs=epochs,
        verbose=1
    )

    # 可视化曲线
    if plot_curves:
        plt.figure()
        plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left'); plt.tight_layout(); plt.show()

        plt.figure()
        plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
        plt.title('Model Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left'); plt.tight_layout(); plt.show()

    # 测试
    probs = model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)
    truth = np.argmax(y_test, axis=1)
    acc = metrics.accuracy_score(truth, pred)
    print(f"[Week4] Test Accuracy: {acc * 100:.2f}%")

    cm = metrics.confusion_matrix(truth, pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LE.classes_)
    disp.plot(xticks_rotation=45); plt.tight_layout(); plt.show()

    # 保存权重
    out = Path(mfcc_dir).parent / "digit_classification.weights.h5"
    model.save_weights(str(out))
    print(f"[Week4] Saved weights to: {out}")

    return model, LE
