from sklearn import svm

# 二分类
# 训练样本特征
X = [[0, 0], [1, 1]]
# 训练样本类别鉴别器
Y = [0, 1]
cls = svm.SVC()
# 训练
cls.fit(X, Y)
# 测试样本数据
test = [[2, 2]]
# 预测
print(cls.predict(test))


# 多分类
# 训练样本特征
X = [[0], [1], [2], [3], [4]]
# 训练样本的类别标签
Y = [0, 1, 2, 3, 4]
# 测试数据集
test = [[1]]
# 选择一对一策略
cls = svm.SVC(decision_function_shape='ovo')
# 训练
cls.fit(X, Y)
# 查看投票函数
dec = cls.decision_function(test)
# 查看筛选函数的大小，可以看到是10， 是因为ovo策略会设计5*4/2=10个分类器，然后找出概率最大的
print(dec.shape)
print(cls.predict(test))
# 选择一对多策略
cls1 = svm.SVC(decision_function_shape='ovr')
cls1.fit(X, Y)
dec1 = cls1.decision_function(test)
print(dec1.shape)
print(cls1.predict(test))