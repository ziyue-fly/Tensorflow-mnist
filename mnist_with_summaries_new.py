# coding=utf-8

import tensorflow as tf
"""
首先载入Tensorflow，并设置训练的最大步数为1000,学习率为0.001,dropout的保留比率为0.9。
同时，设置MNIST数据下载地址data_dir和汇总数据的日志存放路径log_dir。
这里的日志路径log_dir非常重要，会存放所有汇总数据供Tensorflow展示。
"""
from tensorflow.examples.tutorials.mnist import input_data
max_step = 1000  # 最大迭代次数（步数）
learning_rate = 0.001  #学习率
dropout = 0.9  # 保留的数据比率
data_dir = 'E:/学习/大三下/涉密信息系统/实验5 人工智能/TFtest/MNIST_data'
log_dir = 'f:/logs'

# 使用input_data.read_data_sets下载MNIST数据，并创建Tensorflow的默认Session
mnist = input_data.read_data_sets(data_dir, one_hot=True)  # 把y这一列变成one_hot编码
sess = tf.InteractiveSession()

"""
为了在TensorBoard中展示节点名称，设计网络时会常使用tf.name_scope限制命名空间，
在这个with下所有的节点都会自动命名为input/xxx这样的格式。
定义输入x和y的placeholder，并将输入的一维数据变形为28×28的图片存储到另一个tensor，
我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]，
这样就可以使用tf.summary.image将图片数据汇总给TensorBoard展示了。
"""
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')  # 用于计算交叉熵

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# 定义神经网络模型参数的初始化方法，
# 权重依然使用常用的truncated_normal进行初始化，偏置则赋值为0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 初始化所有偏置项（截距）
def bias_variable(shape):  
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义对Variable变量的数据汇总函数用于画图
"""
计算出Variable的mean,stddev,max和min，
对这些标量数据使用tf.summary.scalar进行记录和汇总。
同时，使用tf.summary.histogram直接记录变量var的直方图。
"""
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# 设计一个MLP多层神经网络来训练数据，在每一层中都会对模型参数进行数据汇总。
"""
定一个创建一层神经网络并进行数据汇总的函数nn_layer。
这个函数的输入参数有输入数据input_tensor,输入的维度input_dim,输出的维度output_dim和层名称layer_name，激活函数act则默认使用Relu。
在函数内，显示初始化这层神经网络的权重和偏置，并使用前面定义的variable_summaries对variable进行数据汇总。
然后对输入做矩阵乘法并加上偏置，再将未进行激活的结果使用tf.summary.histogram统计直方图。
同时，在使用激活函数后，再使用tf.summary.histogram统计一次。
"""
def nn_layer(input_tensor, input_dim, output_dim, layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)  # 把权重的各个指标（方差，平均值）进行总结
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases  # 带到激活函数之前的公式
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='actvations') # 运用激活函数 函数里面传函数 高阶函数
        tf.summary.histogram('activations', activations)
        return activations

"""
使用刚定义好的nn_layer创建一层神经网络，输入维度是图片的尺寸（784=28×28），输出的维度是隐藏节点数500.
再创建一个Dropout层，并使用tf.summary.scalar记录keep_prob。然后再使用nn_layer定义神经网络的输出层，激活函数为全等映射，此层暂时不使用softmax,在后面会处理。
"""
hidden1 = nn_layer(x, 784, 500, 'layer1')  # 建立第一层 隐藏层

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)  # 使用dropout函数 保留下来的数据 

# 然后使用nn_layer定义神经网络输出层，其输入维度为上一层隐含节点数500，输出维度为类别数10
# 同时激活函数为全等映射identity，暂时不使用softmax
y1 = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)  # 建立第二层 输出层

"""
这里使用tf.nn.softmax_cross_entropy_with_logits()对前面输出层的结果进行softmax处理并计算交叉熵损失cross_entropy。
计算平均损失，并使用tf.summary.saclar进行统计汇总。
"""
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y)
    #输出层给的结果logits=y 
    #每一行的y是有10个数预测10个值 然后利用这10个值做归一化 然后具备一个概率的含义 第二步计算交叉熵
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)  #  平均损失
tf.summary.scalar('cross_entropy', cross_entropy)

"""
使用Adma优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuray，
再使用tf.summary.scalar对accuracy进行统计汇总。
"""
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # AdamOptimizer比SGD更好一些，下降速度更快，更容易计算局部最优解 ，当数据量大的时候不如SGD
    # learning_rate虽然是固定的，后面会自适应，根据上一次的结果 所以大数据量的话，不如定义好策略，这样省时间
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y1, 1), tf.arg_max(y, 1))
    # 预测值最大的索引 和真实值的索引
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    # true 1 false 0 reduce_mean 是一个比例得到的结果
tf.summary.scalar('accuracy', accuracy)

"""
由于之前定义了非常多的tf.summary的汇总操作，一一执行这些操作态麻烦，
所以这里使用tf.summary.merger_all()直接获取所有汇总操作，以便后面执行。
然后，定义两个tf.summary.FileWrite(文件记录器)在不同的子目录，分别用来存放训练和测试的日志数据。
同时，将Session的计算图sess.graph加入训练过程的记录器，这样在TensorBoard的GRAPHS窗口中就能展示整个计算图的可视化效果。
最后使用tf.global_variables_initializer().run()初始化全部变量。
"""
merged = tf.summary.merge_all()  # 汇总
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)  # 用于存放训练的日志数据
test_writer = tf.summary.FileWriter(log_dir + '/test') # 用于存放测试的日志数据
tf.global_variables_initializer().run()  # 初始化全部向量

"""
定义feed_dict的损失函数。
该函数先判断训练标记，如果训练标记为true,则从mnist.train中获取一个batch的样本，并设置dropout值;
如果训练标记为False，则获取测试数据，并设置keep_prob为1,即等于没有dropout效果。
"""
def feed_dict(train):
    if train:  # 如果是训练的话需要Dropout 
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:  # 测试的时候不要Dropout
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y: ys, keep_prob: k}

# 实际执行具体的训练，测试及日志记录的操作
"""
首先，使用tf.train.Saver()创建模型的保存器。
然后，进入训练的循环中，每隔10步执行一次merged（数据汇总），accuracy（求测试集上的预测准确率）操作，
并使应test_write.add_summary将汇总结果summary和循环步数i写入日志文件;
同时每隔100步，使用tf.RunOption定义Tensorflow运行选项，其中设置trace_level为FULL——TRACE,
并使用tf.RunMetadata()定义Tensorflow运行的元信息，
这样可以记录训练时运算时间和内存占用等方面的信息.
再执行merged数据汇总操作和train_step训练操作，将汇总summary和训练元信息run_metadata添加到train_writer.
平时，则执行merged操作和train_step操作，并添加summary到trian_writer。
所有训练全部结束后，关闭train_writer和test_writer。
"""
saver = tf.train.Saver()  #创建模型保存器
for i in range(max_step):  # max_step=1000,迭代次数
    if i % 10 == 0: # 每执行10次的时候汇总一次信息，然后计算测试集的一次准确率，即10的倍数为测试集
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        # 将所有的日志写入文件，TensorFlow程序就可以那这次运行日志文件，进行各种信息的可视化
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # 训练，即除10的倍数之外都为训练集
        if i % 100 == 99:  # 每到100次记录训练的运算时间和内存占用等方面的信息
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()  
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir+"/model.ckpt", i)
            print('Adding run metadata for', i)
        else:  
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()

