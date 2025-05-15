import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 打印TensorFlow版本
print(f"TensorFlow version: {tf.__version__}")

# 创建一个简单的神经网络模型
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 加载MNIST数据集
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # 将图像展平为一维数组
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    
    return (x_train, y_train), (x_test, y_test)

# 可视化训练历史
def visualize_training_history(history):
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制训练和验证准确率
    ax1.plot(history.history['accuracy'], label='训练准确率')
    ax1.plot(history.history['val_accuracy'], label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制训练和验证损失
    ax2.plot(history.history['loss'], label='训练损失')
    ax2.plot(history.history['val_loss'], label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    # 调整布局并保存图表
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 可视化一些预测结果
def visualize_predictions(model, x_test, y_test, num_samples=5):
    # 随机选择一些测试样本
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # 获取预测结果
    predictions = model.predict(x_test[indices])
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 创建图表
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # 将展平的图像重新变形为28x28
        img = x_test[idx].reshape(28, 28)
        
        # 显示图像
        axes[i].imshow(img, cmap='gray')
        
        # 设置标题显示真实标签和预测标签
        true_label = y_test[idx]
        pred_label = predicted_classes[i]
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}', color=color)
        
        # 移除坐标轴
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

# 主函数
def main():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # 创建模型
    model = create_model()
    
    # 训练模型，使用验证集
    history = model.fit(
        x_train, y_train, 
        epochs=5, 
        batch_size=32, 
        validation_split=0.2,  # 使用20%的训练数据作为验证集
        verbose=1
    )
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # 可视化训练历史
    visualize_training_history(history)
    
    # 可视化一些预测结果
    visualize_predictions(model, x_test, y_test)

if __name__ == "__main__":
    main()
