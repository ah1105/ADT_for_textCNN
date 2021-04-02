# ADT_for_textCNN
adversarial train for textCNN

#环境
在GPU集群训练
python 3.7
pytorch 1.6
tqdm
sklearn

#项目文件介绍
模型文件：
fgm模型：adv_model_fgm
fgsm模型：adv_model_fgsm
free模型：adv_model_free
base模型：adv_model_none
pgd模型：adv_model_pgd

训练脚本：train_ad.py

测试脚本：evaluate_ad.py

训练测试数据：./THUCNews/data

#训练
修改train_ad.py里面attack参数来训练base、fgm、fgsm、pgd、free五种模型

python train_ad.py

#测试
修改evaluate_ad.py里面attack，none不受attack的正常测试，pgd受到PGD的测试方式。可以同时测试五种模型。log里面可以看到测试结果。

python evaluate_ad.py


