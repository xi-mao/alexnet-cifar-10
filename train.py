#coding=utf-8
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time

from alexnet import model  
from data import get_data_set

train_x,train_y,tain_l=get_data_set("train")
test_x,test_y,test_l=get_data_set("test")

x,y,output,global_step,y_pred_cls=model()
_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_ITERATION = 30000
_SAVE_PATH = "tensorboard/cifar-10/"  #先创建好这些文件
_SAVE_BOARD_PATH="tensorboard/board/"

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss,global_step=global_step)

correct_prediction=tf.equal(y_pred_cls,tf.argmax(y,axis=1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.summary.scalar('loss',loss)
tf.summary.scalar("Accyracy/train",accuracy)
tf.summary.histogram('histogram',accuracy)
saver=tf.train.Saver()
sess=tf.Session()
merged=tf.summary.merge_all()

train_writer=tf.summary.FileWriter(_SAVE_BOARD_PATH,sess.graph)
sess.run(tf.global_variables_initializer())

#sess_path=saver.save(sess,_SAVE_PATH)
#try:
#   print("Trying to restore last checkpoint..... ")
#    last_chk_path=tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)   #将变量保存在此路径
#   saver.restore(sess,save_path=last_chk_path)
#    print("Restored checkpoint from:",last_chk_path)
#except:
#    print("Failed to restore checkpoint.Initializing variables instead")
#    sess.run(tf.global_variables_initializer())

def train(num_iterations):
    for i in range(num_iterations):
        randidx=np.random.randint(len(train_x),size=_BATCH_SIZE)    #此处返回的是小于冷（train）的离散均匀分布，总共有128个
        batch_xs=train_x[randidx]
        batch_ys=train_y[randidx]

        start_time=time()
        i_global,_=sess.run([global_step,optimizer],feed_dict={x:batch_xs,y:batch_ys})
        duration=time()-start_time


        if(i_global%10==0)or(i==num_iterations-1):
            _loss,batch_acc=sess.run([loss,accuracy],feed_dict={x:batch_xs,y:batch_ys})
            msg= "Glo bal Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))

            resultmerged=sess.run(merged,feed_dict={x:batch_xs,y:batch_ys})
            train_writer.add_summary(resultmerged,i_global)


        if  (i_global%100==0)or(i==num_iterations-1):
            
            acc=predict_test()

            print('test accuracy is:')
            print(acc)
            saver.save(sess,save_path=_SAVE_PATH,global_step=global_step)
            print("Saved checkpoint")


def predict_test(show_confusion_matrix=False):

    i=0
    predicted_class=np.zeros(shape=len(test_x),dtype=np.int)#返回一个新的数组，用零填充
    print('test_x的长度：')
    print(len(test_x))

    while i<len(test_x):
        j=min(i+_BATCH_SIZE,len(test_x))
        batch_xs=test_x[i:j,:]
        #batch_xs是128*3072的大小，最后一个是16*3072
        batch_ys=test_y[i:j,:]
        predicted_class[i:j]=sess.run(y_pred_cls,feed_dict={x:batch_xs,y:batch_ys})
        i=j

    correct=(np.argmax(test_y,axis=1)==predicted_class)
    acc=correct.mean()*100

    correct_numbers=correct.sum()

    print("Accuracy on Test-Set:{0:.2f}%({1}/{2})".format(acc,correct_numbers,len(test_x)))

    if show_confusion_matrix is True:
        cm=confusion_matrix(y_true=np.argmax(test_y,axis=1),y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name="({}){}".format(i,test_l[i])
            print (cm[i:],class_name)
        class_numbers=["({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))
    
    return acc

if _ITERATION!=0:
    train(_ITERATION)

sess.close()












