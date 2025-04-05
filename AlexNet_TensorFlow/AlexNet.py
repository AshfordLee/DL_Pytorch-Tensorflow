import tensorflow as tf
from d2l import tensorflow as d2l
from tensorflow.keras import layers

def net():
    model=tf.keras.Sequential([
        layers.Conv2D(filters=96,
                      kernel_size=11,
                      strides=4,
                      padding='same'),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=3,
                            strides=2),
        layers.Conv2D(filters=256,
                      kernel_size=5,
                      padding='same'),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=3,
                            strides=2),

        layers.Conv2D(filters=384,
                      kernel_size=3,
                      padding='same'),
        layers.ReLU(),

        layers.Conv2D(filters=256,
                      kernel_size=3,
                      padding='same'),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=3,strides=2),
        layers.Flatten(),
        layers.Dense(4096),
        layers.ReLU(),
        layers.Dropout(0.5),

        layers.Dense(4096),
        layers.ReLU(),
        layers.Dropout(0.5),

        layers.Dense(10)
    ])

    return model


print("TensorFlow版本:",tf.__version__)
print("可用设备")

for device in tf.config.list_physical_devices():
    print("设备名称",device.name,
          "设备类型",device.device_type)
    
gpus=tf.config.list_physical_devices('GPU')

if gpus:
    print(f"找到{len(gpus)}个GPU")
    for gpu in gpus:
        print("GPU名称",gpu.name)

    with tf.device('/GPU:0'):
        x=tf.random.normal([5,3])
        print(x)
        print("使用设备",x.device)

    gpu_device=tf.test.gpu_device_name()
    print("\nGPU设备名称:", gpu_device)
    print("CUDA是否可用:", tf.test.is_built_with_cuda())

else:
    print("\n警告: 未检测到GPU设备，将使用CPU进行计算")
    # 在CPU上创建测试张量
    with tf.device('/CPU:0'):
        x = tf.random.normal([5, 3])
        print("\nCPU张量:")
        print(x)
        print("设备:", x.device)

if gpus:
    try:
        print("\nGPU内存信息:")
        print(tf.config.experimental.get_memory_info('GPU:0'))
    except:
        print("\n无法获取GPU内存信息")



X=tf.random.uniform((1,224,224,1))
model=net()

for layer in model.layers:
    X=layer(X)
    print(layer.__class__.__name__,X.shape)


batch_size=128
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,
                                                 resize=224)

for X, y in train_iter:
    print("训练数据批次形状:", X.shape)
    print("标签批次形状:", y.shape)
    break


lr=0.01
num_epochs=100

d2l.train_ch6(net,
              train_iter=train_iter,
              test_iter=test_iter,
              num_epochs=num_epochs,
              lr=lr,
              device=d2l.try_gpu())