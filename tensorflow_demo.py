import tensorflow as tf

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

# 主函数
def main():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # 创建模型
    model = create_model()
    
    # 训练模型
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    
    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
