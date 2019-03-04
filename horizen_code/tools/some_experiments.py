import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def warmup_and_cosine_lr(init_lr, global_step, warmup_step, decay_steps, alpha=1e-6):
    def warmup_lr(init_lr, global_step, warmup_step):
        global_step = tf.cast(global_step, tf.float32)
        return 1e-6 + (init_lr - 1e-6) * global_step / warmup_step

    def cosine_lr(init_lr, global_step, decay_steps, alpha=0.0):
        return tf.train.cosine_decay(learning_rate=init_lr,
                                     global_step=global_step - warmup_step,
                                     decay_steps=decay_steps - warmup_step,
                                     alpha=alpha)

    return tf.cond(tf.less_equal(global_step, warmup_step),
                   true_fn=lambda: warmup_lr(init_lr, global_step, warmup_step),
                   false_fn=lambda: cosine_lr(init_lr, global_step, decay_steps, alpha))
def test():

    x = np.arange(400000)
    gl_placeholder = tf.placeholder(dtype=tf.int64, shape=None)
    cosine_lr = warmup_and_cosine_lr(init_lr=1e-3,
                                  global_step=gl_placeholder,
                                  warmup_step=20000,
                                  decay_steps=400000,
                                  alpha=1e-6)
    constant_lr = tf.train.piecewise_constant(gl_placeholder,
                                     boundaries=[220000, 320000],
                                     values=[1e-3, 1e-4, 1e-5])
    cosine_lr_list = []
    constant_lr_list = []

    with tf.Session() as sess:
        for i in range(400000):
            i = int(i)
            feed_dict = {gl_placeholder: i}
            cosine_lr_, constant_lr_ = sess.run([cosine_lr, constant_lr], feed_dict=feed_dict)
            # lr_list.append(lr_)
            cosine_lr_list.append(cosine_lr_)
            constant_lr_list.append(constant_lr_)
            if i %10000 == 0:
                print i
    print(len(cosine_lr_list))
    np.save('cos_lr.npy', np.array(cosine_lr_list))
    np.save('constatn_lr.npy', np.array(constant_lr_list))
    plt.plot(x, constant_lr_list, 'r')
    plt.plot(x, cosine_lr_list)
    plt.show()

if __name__ == '__main__':
    test()