mnist_uint8.mat 是MNIST的mat版，x为手写图片（拉成一维），y为one-hot形式的标签
包含train_x（60000*784 uint8）, train_y（60000*10 uint8）, test_x（10000*784 uint8）, test_y（10000*10 uint8）,
其中train_x和test_x 的元素为【0,255】， train_y和test_y的元素为【0,1】（one-hot）

MNIST_diff 是由MNIST数据集生成的差分加噪阈值化数据集（模拟事件序列映射为二维的效果），_diff为加噪数据集，_diff_gt为不加噪的ground truth
包含train_diff（60000*28*28 double）, train_diff_gt（60000*28*28 double）, test_diff（10000*28*28 double）, test_diff_gt（10000*28*28 double）,
所有元素均为0,-1,或+1
（差分方法为先移动再差分，其中train数据中，四个水平方向各有10000个样本移动2像素，四个对角线方向各有5000个样本移动（1,1）像素。
test数据中，四个水平方向各有1250个样本移动2像素、四个对角线方向各有1250个样本移动（1,1）像素）

MNIST_diff_labels是原MNIST的标签，one-hot类型，包含train_labels（60000*10 double），test_labels（10000*10 double）

MNIST2evt为将mnist_uint8转换为MNIST_diff_train和MNIST_diff_test的matlab源程序

load_mat为从python加噪mat数据并进行批量化的测试代码（相关函数已经写入自定义python库my_tf_lib\my_io，可直接导入调用）

