# ConvDAE：convolutional denoising autoencoder
By：zhihong

Email: zzh.ll@outlook.com



directories:

* logs：训练中损失函数变化曲线记录
* model_data：训练完成的模型保存文件夹
* MNIST_diff：数据集1
* N_MNIST_pic：数据集2
* predict_res：预测结果保存文件夹（需要在 pred代码中 将saveflag = 1）
* L-ConvDAE, U-ConvDAE, ConvDAE_unsup：三种类型的卷积去噪自编码器网络模型
* ConvDAE_pred：加载现有模型进行预测，需要根据网络结构进行修改
* ConvDAE_finetune：模型微调代码，需要根据网络结构进行修改