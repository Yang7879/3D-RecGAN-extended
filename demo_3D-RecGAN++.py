import os
import numpy as np
import scipy.io
import tensorflow as tf
import tools

GPU0 = '0'

def ttest_demo():
    ####### load sample data
    x_path = './Data_sample/P1_03001627_chair/test_25d_vox256/1c08f2aa305f124262e682c9809bff14_0_0_0.npz'
    y_true_path = './Data_sample/P1_03001627_chair/test_3d_vox256/1c08f2aa305f124262e682c9809bff14_0_0_0.npz'
    x_sample = tools.Data.load_single_voxel_grid(x_path, out_vox_res=64)
    y_true = tools.Data.load_single_voxel_grid(y_true_path, out_vox_res=256)

    ####### load model + testing
    model_path = './Model_released/'
    if not os.path.isfile(model_path + 'model.cptk.data-00000-of-00001'):
        print ('please download our released model first!')
        return

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = GPU0
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph( model_path +'model.cptk.meta', clear_devices=True)
        saver.restore(sess, model_path+ 'model.cptk')
        print ('model restored!')

        X = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        Y_pred = tf.get_default_graph().get_tensor_by_name("aeu/Sigmoid:0")
        x_sample = x_sample.reshape(1, 64, 64, 64, 1)
        y_pred = sess.run(Y_pred, feed_dict={X: x_sample})

    ###### save result
    x_sample = x_sample.reshape(64, 64, 64)
    y_pred = y_pred.reshape(256, 256, 256)
    x_sample = x_sample.astype(np.int8)
    y_pred = y_pred.astype(np.float16)
    y_true = y_true.astype(np.int8)
    to_save = {'X_test': x_sample, 'Y_test_pred': y_pred, 'Y_test_true': y_true}
    scipy.io.savemat('demo_result.mat', to_save, do_compression=True)
    print ('results saved.')

def visualize():
    ######
    result_path = 'demo_result.mat'
    mat = scipy.io.loadmat(result_path)
    x_sample = mat['X_test']
    y_pred = mat['Y_test_pred']
    y_true = mat['Y_test_true']

    ######  if the GPU serve is able to visualize, otherwise comment the following lines
    th = 0.5
    y_pred[y_pred >= th] = 1
    y_pred[y_pred < th] = 0
    tools.Data.plotFromVoxels(x_sample, title='x_sample')
    tools.Data.plotFromVoxels(y_pred, title='y_pred')
    tools.Data.plotFromVoxels(y_true, title='y_true')
    from matplotlib.pyplot import show
    show()

#########################
if __name__ == '__main__':
    ttest_demo()
    visualize()