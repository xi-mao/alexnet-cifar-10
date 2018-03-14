#coding=utf-8
import math
import tensorflow as tf

def print_activations(t):
    print(t.op.name,'',t.get_shape().as_list)   #get_shape获取一个TensorShape对象，然后通过as_list方法返回每一个维度数

def model():
    _IMAGE_SIZE=32
    _IMAGE_CHANNELS=3
    _RESHAPE_SIZE=3*3*128
    _NUM_CLASSES=10

    parameters=[]
    with  tf.name_scope('data'):
        x=tf.placeholder(tf.float32,shape=[None,_IMAGE_SIZE*_IMAGE_SIZE*_IMAGE_CHANNELS],name='images')
        y=tf.placeholder(tf.float32,shape=[None,_NUM_CLASSES],name='Output')
        images=tf.reshape(x,[-1,_IMAGE_SIZE,_IMAGE_SIZE,_IMAGE_CHANNELS],name='images')
        print(images) 
    #conv1
    #这里name_scope实际上是为了解决共享变量的问题，在name_scope下进行tf.Variable(name)
    #如果name重名，会自动检测命名冲突进行处理   
    with tf.name_scope('conv1') as scope:          
        kernel=tf.Variable(tf.truncated_normal([5,5,3,64],dtype=tf.float32,
                                        stddev=1e-1),name='weights')
        #变量解释 [a,b,c,d]分别表示,1表示是否跳过一些样本，比如a=1时，就是从1，2，3...训
        #跳过一些，a=2时就选择1，3，5...，b表示高方向滑动，c表示宽方向滑动，d表示通道滑动
        #same表示当卷积核超出边界时会进行0填充
        conv=tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
                                        trainable=True,name='bias')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)    #这里返回的是一个tensor（一个张量类），但是这里的name=scope是什么意思？
        print_activations(conv1)
    tf.summary.histogram('Convolution_layers/conv1',conv1)
    tf.summary.scalar('Convolution_layers/conver1',tf.nn.zero_fraction(conv1))

    #这一步时local Response Normalization技术详情可以查看论文中描述
    #lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1=tf.nn.local_response_normalization(conv1,
                                                alpha=1e-4,
                                                beta=0.75,
                                                depth_radius=2,
                                                bias=2.0)
    #pool1
    pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1], 
                    padding='VALID',name='pool1')

    print_activations(pool1)

    #conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                                 trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
    tf.summary.histogram('Convolution_layers/conv2',conv2)
    tf.summary.scalar('Convolution_layers/conver2',tf.nn.zero_fraction(conv2))
    print_activations(conv2)
    #lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.local_response_normalization(conv2,alpha=1e-4,beta=0.75,
                                                depth_radius=2, bias=2.0)
    # pool2
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],
                            padding='VALID',name='pool2')
    print_activations(pool2)

    #conv3
    with tf.name_scope('conv3') as scope:
        kernel =tf.Variable(tf.truncated_normal([3,3,64,128],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32),
                                        trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        print_activations(conv3)
    tf.summary.histogram('Convolution_layers/conv3',conv3)
    tf.summary.scalar('Convolution_layers/conver3',tf.nn.zero_fraction(conv3))
    #conv4
    with tf.name_scope('conv4') as scope:
        kernel =tf.Variable(tf.truncated_normal([3,3,128,128],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32),
                                        trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)
        print_activations(conv4)
    tf.summary.histogram('Convolution_layers/conv4',conv4)
    tf.summary.scalar('Convolution_layers/conver4',tf.nn.zero_fraction(conv4))
    #conv5
    with tf.name_scope('conv5') as scope:
        kernel =tf.Variable(tf.truncated_normal([3,3,128,128],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[128],dtype=tf.float32),
                                        trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)
        print_activations(conv5)
    tf.summary.histogram('Convolution_layers/conv5',conv5)
    tf.summary.scalar('Convolution_layers/conver5',tf.nn.zero_fraction(conv5))

    #pool5
    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],
                            padding='VALID',name='pool5')
    print_activations(pool5)

    #fully_connected1
    with tf.name_scope('fully_connected1') as scope:
        reshape=tf.reshape(pool5,[-1,_RESHAPE_SIZE])
        dim=reshape.get_shape()[1].value
        weights =tf.Variable(tf.truncated_normal([dim,384],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        print_activations(weights)
        biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),
                                        trainable=True,name='biases')
        local3=tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope)
        print_activations(local3)
    tf.summary.histogram('Fully connected layers/fc1',local3)
    tf.summary.scalar('Fully connected layers/fc1',tf.nn.zero_fraction(local3))

    #fully_connected2
    with tf.name_scope('fully_connected') as scope:
        weights =tf.Variable(tf.truncated_normal([384,192],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        print_activations(weights)
        biases=tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),
                                        trainable=True,name='biases')
        local4=tf.nn.relu(tf.matmul(local3,weights)+biases,name=scope)
        print_activations(local4)
    tf.summary.histogram('Fully connected layers/fc2',local4)
    tf.summary.scalar('Fully connected layers/fc4',tf.nn.zero_fraction(local4))

    #output
    with tf.name_scope('output') as scope:
        weights =tf.Variable(tf.truncated_normal([192,_NUM_CLASSES],dtype=tf.float32,
                                                stddev=1e-1),name='weights')
        print_activations(weights)
        biases=tf.Variable(tf.constant(0.0,shape=[_NUM_CLASSES],dtype=tf.float32),
                                        trainable=True,name='biases')
        softmax_linear=tf.add(tf.matmul(local4,weights),biases,name=scope)
    tf.summary.histogram('Fully connected layers/output',softmax_linear)

    global_step=tf.Variable(initial_value=0,name='global_step',trainable=False)
    y_pred_cls=tf.argmax(softmax_linear,axis=1)


    return x,y,softmax_linear,global_step,y_pred_cls



