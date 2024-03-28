#情感分析-数据预处理
import os
import torch
from torch import nn
import dltools

def read_imdb(data_dir,is_train): #读取IMDB评论数据集的 文本序列和标签（是否是训练数据）
    data,lables=[],[]
    for lable in ('pos','neg'):
        # 根据is_train决定是否用训练数据,并分为pos和neg两部分
        folder_name=os.path.join(data_dir,'train' if is_train else 'test',lable)
        folder_name=folder_name.replace('\\', '/')
        for file in os.listdir(folder_name): #返回指定目录下的所有文件和文件夹的名称（每一个txt是一条单独的评论）
            with open(os.path.join(folder_name,file),'rb') as f:
                review=f.read().decode('utf-8').replace('\n','') #replace:去除换行符
                data.append(review)
                lables.append(1 if lable=='pos' else 0)
    return data,lables # 返回文本内容，标签（见x,y）

data_dir='D:/桌面/work_old/code/test/aclImdb/'
train_data=read_imdb(data_dir,is_train=True)
# 训练集数目是25000（pos+neg）
print('训练集数目：',len(train_data[0]))
for x,y in zip(train_data[0][:3],train_data[1][:3]): # 文本内容 标签
    print('标签：',y,'review：',x[0:60])
    # 标签： 1 review： Bromwell High is a cartoon comedy. It ran at the same time a
    # 标签： 1 review： Homelessness (or Houselessness as George Carlin stated) has
    # 标签： 1 review： Brilliant over-acting by Lesley Ann Warren. Best dramatic ho

# 数据预处理
train_tokens=dltools.tokenize(train_data[0],token='word') # token化，切分成单词
vocab=dltools.Vocab(train_tokens,min_freq=5,reserved_tokens=['<pad>']) # 以词频为5做过滤，再做长度统一

num_step=500 # 序列长度，每行500（按评论字数决定，以500字做分割）
# 创建张量（tensor）,类似NumPy中的数组，但能够在GPU上运行并加速数值计算
# 截断(truncation)和 填充(padding)，不足的用vocab['<pad>']（1）来填充，保证25000行每行都是500
# vocab将单词映射成数字。train_tokens（字符串列表）->train_features（数字列表，特征）
train_features=torch.tensor(
    [dltools.truncate_pad(vocab[line],num_step,vocab['<pad>']) for line in train_tokens])

# 创建训练用数据集
# (data和lable) train_features是等量切分的（数字特征），torch.tensor(train_data[1])是对应标签
# train_features的torch.Size([25000, 500])，25000条文本数据，每条500个数（特征）
# train_data是25000个元组。train_data[1]是25000条文本的标签。
# 每个batch的大小64
train_iter=dltools.load_array((train_features,torch.tensor(train_data[1])),64)

# for x,y in train_iter:
#     print('x:',x.shape,'y:',y.shape)
#     break
# print('小批量数目：',len(train_iter))
# x:torch.size([64,500]),y:torch.size([64])
# 每个小批量里64条数据，结构是x:64条长度500的feature,以及y:64条数据的标签
# 小批量数目：391

def load_data_imdb(batch_size,num_steps=500):
    # 返回数据迭代器和IMDb评论数据集的词表
    # data_dir=dltools.download_extract('aclImdb','aclImdb') #名称，文件夹
    data_dir = 'D:/桌面/work_old/code/test/aclImdb/'
    train_data=read_imdb(data_dir,True)
    test_data=read_imdb(data_dir,False)
    train_tokens=dltools.tokenize(train_data[0],token='word')
    test_tokens=dltools.tokenize(test_data[0],token='word')
    vocab=dltools.Vocab(train_tokens,min_freq=5)
    train_features=torch.tensor(
        [dltools.truncate_pad(vocab[line],num_steps,vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor(
        [dltools.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter=dltools.load_array((train_features,torch.tensor(train_data[1])),batch_size)
    test_iter=dltools.load_array((test_features,torch.tensor(test_data[1])),batch_size,is_train=False)
    return train_iter,test_iter,vocab

batch_size = 64
train_iter, test_iter, vocab = dltools.load_data_imdb(batch_size) # 返回数据迭代器和词表



# 使用循环神经网络表示单个文本
class BiRNN(nn.Module):
    # BiRNN的父类是nn.Module
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        # **kwargs：①特殊语法，定义函数时接受不定数量的关键字参数（keyword arguments），并将这些参数封装成一个字典。
        # ②将接受到的关键字参数传递给其他函数（如传递给父类）

        # super函数通过方法解析顺序（MRO）来确定应该调用哪个父类的方法或构造函数。
        # 调用BiRNN类的父类nn.Module的构造函数，并将传递给子类的任何关键字参数传递给父类。

        # super函数接受两个参数：
        # 第一个参数是子类的类名，用于指定要调用哪个父类的构造函数。
        # 第二个参数是子类的实例对象，用于传递给父类构造函数的 self 参数。
        super(BiRNN, self).__init__(**kwargs)

        # 输入参数(词汇表大小/词语数目，词向量维度)。
        # 将输入的词语索引转换为词向量，创建一个可训练的词嵌入矩阵，其中每行表示一个词语的词向量。
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # (词向量/输入维度，隐藏状态维度，LSTM层的数量，bidirectional为True获取双向循环神经网络)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,bidirectional=True)
        # 将输入的特征向量映射成预测的类别（2个）
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs：（批量大小，时间步数）
        # 输出：（时间步数，批量大小，词向量维度）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，需要转置。
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters() # 优化了一下性能

        # 两个返回参数分别是，上一个隐藏层在所有时间步的隐状态，和在最后一个时间步的隐状态
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数），所以decode的时候要4个隐藏单元数
        # 把embedding传给encoder
        outputs, _ = self.encoder(embeddings)

        # cat连结开始的和最终的时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs

# 定义词向量维度/输入维度，隐藏层维度，隐藏层数量
embed_size, num_hiddens, num_layers = 100, 100, 2
devices = dltools.try_all_gpus() # 尝试查找可用gpu设备，没有的话在cpu上跑
# 实例化net
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) # 均匀分布
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names: # 获取参数名称
            if "weight" in param: # 对包含"weight"关键字的参数名称
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights); # 将init_weights函数应用到神经网络net的所有子模块上，完成参数初始化的操作

# 加载了预训练的GloVe词向量模型
glove_embedding = dltools.TokenEmbedding('glove.6b.100d') # 降低过拟合

# vocab可以用预训练的词向量表示（vocab.idx_to_token：'<unk>', 'the', 'a', 'and', 'of',……）
embeds = glove_embedding[vocab.idx_to_token]
# embeds.size [49346,100]

# net的嵌入层的权重矩阵
net.embedding.weight.data.copy_(embeds) # copy，将embeds直接应用到net.embedding.weight
net.embedding.weight.requires_grad = False # 不用计算梯度

# 训练和模型评估
lr, num_epochs = 0.01, 5 # 学习率和训练迭代次数
# 创建一个Adam 优化器（optimizer），用于优化神经网络模型net的参数。（此处是所有参数）
trainer = torch.optim.Adam(net.parameters(), lr=lr)
# 交叉熵损失函数，分别计算每个样本的损失，而不进行任何计算。
loss = nn.CrossEntropyLoss(reduction="none")
# 训练
dltools.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,devices)

# 预测环节
def predict_sentiment(net, vocab, sequence):
    """预测文本序列的情感"""
    sequence = torch.tensor(vocab[sequence.split()], device=dltools.try_gpu())
    # 将sequence传入net，用argmax求出对应的label
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

predict_sentiment(net, vocab, 'this movie is so great')
predict_sentiment(net, vocab, 'this movie is worse')