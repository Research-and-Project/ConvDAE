mnist_uint8.mat ��MNIST��mat�棬xΪ��дͼƬ������һά����yΪone-hot��ʽ�ı�ǩ
����train_x��60000*784 uint8��, train_y��60000*10 uint8��, test_x��10000*784 uint8��, test_y��10000*10 uint8��,
����train_x��test_x ��Ԫ��Ϊ��0,255���� train_y��test_y��Ԫ��Ϊ��0,1����one-hot��

MNIST_diff ����MNIST���ݼ����ɵĲ�ּ�����ֵ�����ݼ���ģ���¼�����ӳ��Ϊ��ά��Ч������_diffΪ�������ݼ���_diff_gtΪ�������ground truth
����train_diff��60000*28*28 double��, train_diff_gt��60000*28*28 double��, test_diff��10000*28*28 double��, test_diff_gt��10000*28*28 double��,
����Ԫ�ؾ�Ϊ0,-1,��+1
����ַ���Ϊ���ƶ��ٲ�֣�����train�����У��ĸ�ˮƽ�������10000�������ƶ�2���أ��ĸ��Խ��߷������5000�������ƶ���1,1�����ء�
test�����У��ĸ�ˮƽ�������1250�������ƶ�2���ء��ĸ��Խ��߷������1250�������ƶ���1,1�����أ�

MNIST_diff_labels��ԭMNIST�ı�ǩ��one-hot���ͣ�����train_labels��60000*10 double����test_labels��10000*10 double��

MNIST2evtΪ��mnist_uint8ת��ΪMNIST_diff_train��MNIST_diff_test��matlabԴ����

load_matΪ��python����mat���ݲ������������Ĳ��Դ��루��غ����Ѿ�д���Զ���python��my_tf_lib\my_io����ֱ�ӵ�����ã�

