import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Hyper-parameters
input_size = 2
num_classes = 1
num_epochs = 1000
batch_size = 100
learning_rate = 0.01

# 1️⃣ 生成二维数据
X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)

# 转为 tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2️⃣ DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 3️⃣ 模型（二维输入 → 1输出）
model = nn.Linear(input_size, num_classes)

# 4️⃣ Loss 和 optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 5️⃣ 训练
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):

        # Forward
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6️⃣ 测试
with torch.no_grad():
    correct = 0
    total = 0

    for features, labels in test_loader:
        outputs = model(features)
        predicted = torch.sigmoid(outputs) > 0.5  # 二分类阈值
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy: {:.2f}%'.format(100 * correct / total))

# 7️⃣ 保存模型
torch.save(model.state_dict(), 'binary_model.ckpt')