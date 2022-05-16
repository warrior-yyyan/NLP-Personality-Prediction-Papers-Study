[TOC]

# 1. A sentiment-aware deep learning approach for personality detection from text

[paper](E:\Five Factor Research\finished\A sentiment-aware deep learning approach for personality detection from text.pdf)

- 采用1999 essay Big Five dataset 和 kaggle MBTI dataset
- BERT + CNN / GRU / LSTM结构，加入了Sentic情感词典信息
- 交叉熵损失
- 对MBTI倾斜数据集做了处理
- 采用sentence-level编码，把每一句输入BERT, 然后取倒数第二层的CLS作为表示，将一个人的所有句子拿CLS拼接起来做预测
- 对Sentic的词从-1 ~ 1映射为了0 ~ 2 且每0.1是一个level  映射为20个level，然后变成一个20 dimension的 one-hot vector 拼接进BERT中的768维中变成 788维
- 引入了超参数K，表示为连续K个句子输入进模型的效果，后续实验给出了最佳的K，和在最佳的K上做的实验结果
- 引入皮威尔逊系数说明了各种人格之间的联系不大
- 对比了单标签分类和多标签分类



# 2.Who Am I? Personality Detection Based on Deep Learning for Texts

[paper](E:\Five Factor Research\finished\Who Am I Personality Detection Based on Deep Learning for Texts.pdf)

- 把word的vector拼接在双向LSTM向量中间，形成了 LL-word-LR的形式
- 输入长度为500个word的序列用maxpool降为25长度，25代表25个句子，划分为句子级别
- 接下来沿用textCNN的思想，3个不同的kernel做句子级别的特征提取，然后maxpool，全连接降维分类

# 3. Predicting Myers-Briggs Type....

[paper](E:\Five Factor Research\finished\Predicting Myers-Briggs Type.....pdf)

- kaggle中的数据比例失衡，在测试时人为的调整了各种性格的比例，达到正常比例
- 用NLTK处理数据，去掉网站链接，去掉停用词，去掉正文中含有INTJ INFP等预测名词，以防止网络学到记住这些知识
- 规定每句话长度为40词
- Embedding用的Glove，50 dimension
- 用的RNN，LSTM效果最好，加了0.1的dropout
- 训练4个二分类器
- 对用户人格预测的两种方式
  - 第一，把这个用户的所有句子都做预测，去类别最多的那个类
  - 第二，取平均（取平均的效果好很多）
- 16分类可能会在绝对准确度上比较高，但是这并没有考虑性格之间的关联，4个二分类可以得到较高的近似正确率



# 4. Myers-Briggs Personality Classiﬁcation and Personality-Speciﬁc Language Generation Using Pre-trained Language Models

[paper](E:\Five Factor Research\finished\Myers-Briggs Personality Classiﬁcation and Personality-Speciﬁc.pdf)

- 自己爬的数据集
- 清洗数据的时候把MBTI的类型换成了 '<type>'
- 用了BertForSequenceClassiﬁcation做分类，用了bert-base-uncased, （说用large效果会更好）
- 用了一种新的评价指标，at least match
- 16分类 acc做到0.479
- 4个二分类能做到最高0.758



# 5. Deep Learning-Based Document Modeling for Personality Detection from Text

[paper](E:\Five Factor Research\finished\deep-learning-based-personality-detection.pdf)

- 





# 6. Personality Trait Detection Using Bagged SVM over BERT Word Embedding Ensembles

[paper](E:\Five Factor Research\finished\Personality Trait Detection Using Bagged SVM over BERT Word Embedding Ensembles.pdf)

- Bert最长输入为512， 文章平均长度为650词，本文采用了sub-doc的方法，把文章切分为200词的小文章的组合，然后给这些小文章和大文章一样的标签
- 把文章按照句号 问号处切分为句子组合，并删除所有ASCII 字母、数字、引号和感叹号以外的字符
- 把缩写 you're 变成you are等，把sub-doc的长度从200变为250
- 把bert后四层向量和Mairesse特征拼接起来，形成3156维向量
- 使用10个SVM做分类，类似于bagging classifier
- BB-SVM模型平均预测精度 59.03
- 用同样的分类器，不同的Word Embedding 做了对比，取后四层的Bert向量 好过 w2v 将近2个点
- 取BERT最后四层表示 好过 取任何一层表示（单层对比来看9,10,11,12最后四层好过其它层）





# 7. Bottom-Up and Top-Down: Predicting Personality with Psycholinguistic and Language Model Features
[paper](E:\Personality Prediction\finished\Bottom-Up_and_Top-Down_Predicting_Personality_with_Psycholinguistic_and_Language_Model_Features.pdf)


















