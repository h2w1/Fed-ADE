# -----------------------------------------------------------------------------
# Runtime defaults. These values are overridden by run_experiment.py via globals().
# -----------------------------------------------------------------------------
now_running = globals().get("now_running", 0)
shift = globals().get("shift", "lin")
dist_type = globals().get("dist_type", "uniform")
dist = globals().get("dist", dist_type)
lamda = globals().get("lamda", 300)
T_OVERRIDE = globals().get("T", None)
ROUNDS_OVERRIDE = globals().get("ROUNDS", None)
NUM_CLIENT_OVERRIDE = globals().get("NUM_CLIENT", None)
CUDA_VISIBLE_DEVICES_OVERRIDE = globals().get("cuda_visible_devices", None)
if CUDA_VISIBLE_DEVICES_OVERRIDE is not None:
    import os as _fedade_os
    _fedade_os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES_OVERRIDE)
# -----------------------------------------------------------------------------

# %% [cell 0]
#라이브러리 import
import numpy as np
import matplotlib.pylab as plt
import torchvision
import torch
from torch import nn, optim
import os
from contextlib import contextmanager
from torch.utils.data import TensorDataset ,DataLoader
from torchvision import datasets
import time
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import time
from sklearn.model_selection import train_test_split

import random

from tqdm import tqdm

import gc

device='cuda' if torch.cuda.is_available() else 'cpu'

gc.collect()
torch.cuda.empty_cache()
##########################
# randon seed simulation
# done seed:
#now_running=1
LR_MAX = globals().get("LR_MAX", 1e-3)
LR_MIN = globals().get("LR_MIN", 1e-05)
lamda = globals().get("lamda", 300)
#shift='ber'
#dist_type='uniform'
save_path = f"./model/pretrained_{dist_type}_CIFAR10.pth"
alpha=0.1

random_seed_list=[42,153,248,300,439,517,694,752,846]
RANDOM_SEED=random_seed_list[now_running]
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
opt='Adam'
NUM_CLASSES = 10
C=0.1
NUM_CLIENT = globals().get("NUM_CLIENT", 100)
P_C=int(NUM_CLIENT*C)



#================================
# Pre-training setting
LEARNING_RATE_Pr = 0.0001
BATCH_SIZE_Pr =512
NUM_EPOCHS_Pr =100
IF=1
#================================
# Post-training setting
LEARNING_RATE_Po = 0.0001
BATCH_SIZE_Po = 25
# alpha =0.1


#================================
# Local training
NUM_EPOCHS_LOCAL_Po =5
#================================
# FL
NUM_EPOCHS_FL_Po = 2
ROUNDS = globals().get("ROUNDS", 10)
shared_layers = ['layer1','layer2','layer3']
#================================
# FL_LossFIM
NUM_EPOCHS_LossFIM_P = 2
NUM_EPOCHS_LossFIM_SP= 3
#lamda =100
#LR_MIN = 0.0001
#LR_MAX = 0.001
lamda = globals().get("lamda", 300)
num_test_sample=50

#================================
# Online learning
#shift = "ber"
T = globals().get("T", 100)  # 총 시간 스텝 수
sample_per_step = globals().get("sample_per_step", 50)  # 시간마다 클라이언트에게 제공할 샘플 수
CR=np.arange(ROUNDS)
n_c=list(range(NUM_CLIENT))
p_c=int(len(n_c)*C)
np.random.seed(RANDOM_SEED)
random_list=np.random.randint(low=1, high=1000, size=ROUNDS)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])])

trainset = datasets.CIFAR10(root='data',
                               train=True,
                               transform=transform,
                               download=True)
testset = datasets.CIFAR10(root='data',
                              train=False,
                              transform=transform)

trainloader = DataLoader(testset, batch_size=BATCH_SIZE_Po, shuffle=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE_Po, shuffle=False)

def torch_tensor(x,y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    y = y.type(torch.LongTensor)
    x = x.type(torch.FloatTensor)
    return x,y
X_train, y_train = trainset.data, trainset.targets
X_test, y_test = testset.data, testset.targets
X_train = X_train.astype('float32')
y_train = np.array(y_train)
X_test = X_test.astype('float32')
y_test = np.array(y_test)

X_pre, X_post, y_pre, y_post = train_test_split(X_train, y_train, test_size=1/5, random_state=RANDOM_SEED,stratify=y_train)
X_t, X_post_test, y_t, y_post_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_SEED,stratify=y_test)

X_pre = np.transpose(X_pre, (0, 3, 1, 2))
X_post = np.transpose(X_post, (0, 3, 1, 2))
X_t = np.transpose(X_t, (0, 3, 1, 2))
X_post_test = np.transpose(X_post_test, (0, 3, 1, 2))
num_pre_dataset = len(y_pre)
# 유니폼 테스트셋 생성
# 테스트셋 평가 시 samples_per_class=1000으로 설정
def create_uniform_testset(X_test, y_test, samples_per_class=1000, random_seed=RANDOM_SEED):
    np.random.seed(random_seed)
    selected_indices = []

    for class_label in range(NUM_CLASSES):
        class_indices = np.where(y_test == class_label)[0]
        np.random.shuffle(class_indices)
        selected = class_indices[:samples_per_class]  # 클래스당 n개 선택
        selected_indices.extend(selected)

    selected_indices = np.array(selected_indices)

    X_uniform = X_test[selected_indices]
    y_uniform = y_test[selected_indices]

    return X_uniform, y_uniform


# 유니폼 테스트셋 생성 (예: 클래스당 100개 → 총 1,000개)
X_uniform_test, y_uniform_test = create_uniform_testset(X_test, y_test, samples_per_class=50)

# 텐서 변환
x_uniform_tensor, y_uniform_tensor = torch_tensor(X_uniform_test, y_uniform_test)

# DataLoader 생성
uniform_test_loader = DataLoader(
    TensorDataset(x_uniform_tensor, y_uniform_tensor),
    batch_size=BATCH_SIZE_Po,
    shuffle=False
)
md
### Pretrain용 데이터 분배: longtail, IF1.0: balance, 커질수록 longtail 극심
def create_longtail_dataset(X, y, imbalance_factor=0.1, random_seed=RANDOM_SEED):
    np.random.seed(random_seed)

    # y가 torch.Tensor라면 numpy array로 변환
    if torch.is_tensor(y):
        y_np = y.cpu().numpy()
    else:
        y_np = y

    classes = np.unique(y_np)
    num_classes = len(classes)

    # 각 클래스의 인덱스를 저장할 리스트
    new_indices = []

    # 균형 상태에서 각 클래스의 최대 샘플 수 (일반적으로 모든 클래스가 균등하므로)
    counts = []
    for c in classes:
        indices_c = np.where(y_np == c)[0]
        counts.append(len(indices_c))
    max_count = max(counts)

    # 각 클래스별 long-tail 샘플 수 결정
    # 공식: num_samples = max_count * (imbalance_factor ** (i / (num_classes - 1)))
    # (클래스 번호가 커질수록 샘플 수가 적어짐)
    for i, c in enumerate(classes):
        indices_c = np.where(y_np == c)[0]
        np.random.shuffle(indices_c)
        if num_classes > 1:
            num_samples = int(max_count * (imbalance_factor ** (i / (num_classes - 1))))
        else:
            num_samples = len(indices_c)
        num_samples = max(num_samples, 1)  # 각 클래스는 최소 1개 이상 포함

        selected = indices_c[:num_samples]
        new_indices.extend(selected)

    new_indices = np.array(new_indices)

    # X와 y에서 선택된 인덱스만 추출
    if torch.is_tensor(X):
        X_new = X[new_indices]
    else:
        X_new = X[new_indices]

    if torch.is_tensor(y):
        y_new = y[new_indices]
    else:
        y_new = y[new_indices]

    return X_new, y_new
X_pre, y_pre= torch_tensor(X_pre,y_pre)
X_test, y_test= torch_tensor(X_test,y_test)

X_pre_lt, y_pre_lt = create_longtail_dataset(X_pre, y_pre, imbalance_factor=1/IF, random_seed=RANDOM_SEED)
X_pre_lt_test, y_pre_lt_test = create_longtail_dataset(X_test, y_test, imbalance_factor=1/IF, random_seed=RANDOM_SEED)
# plt.hist(y_pre_lt)
Pre_train=TensorDataset(X_pre_lt, y_pre_lt)
Pre_train_loader=DataLoader(Pre_train,batch_size=BATCH_SIZE_Pr, shuffle=True)
Pre_test=TensorDataset(X_pre_lt_test, y_pre_lt_test)
Pre_test_loader=DataLoader(Pre_test,batch_size=BATCH_SIZE_Pr, shuffle=True)
md
### Post train용 데이터 분배: default: 클라이언트 100, 디리클레 0.1
split_map = dict()
test_split_map = dict()
idx_batch = [[] for _ in range(NUM_CLIENT)]
idxt_batch = [[] for _ in range(NUM_CLIENT)]
np.random.seed(RANDOM_SEED)
min_sample = 6
min_size = 0

while min_size < min_sample:
    idx_batch = [[] for _ in range(NUM_CLIENT)]
    idxt_batch = [[] for _ in range(NUM_CLIENT)]

    for c in range(10):
        idx_k = np.where(y_post == c)[0]  #조건 만족하는 인덱스 찾기# get corresponding class indices
        idx_kt = np.where(y_post_test == c)[0]
        np.random.shuffle(idx_k)  # shuffle class indices
        np.random.shuffle(idx_kt)
        # get label retrieval probability per each client based on a Dirichlet distribution
        proportion = np.random.dirichlet(np.repeat(alpha, NUM_CLIENT))
        proportions = np.array(
            [proportion * (len(idx_j) < len(y_post) / NUM_CLIENT) for proportion, idx_j in zip(proportion, idx_batch)])
        proportionst = np.array(
            [proportion * (len(idx_j) < len(y_post_test) / NUM_CLIENT) for proportion, idx_j in zip(proportion, idxt_batch)])
        #print("proportions2: ", proportion)
        # normalize
        proportions = proportions / proportions.sum()
        proportionst = proportionst / proportionst.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        proportionst = (np.cumsum(proportionst) * len(idx_kt)).astype(int)[:-1]
        # split class indices by proportions
        idx_split = np.array_split(idx_k, proportions)
        idxt_split = np.array_split(idx_kt, proportionst)
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]
        idxt_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idxt_batch, idxt_split)]
        min_size = min([len(idx_j) for idx_j in idx_batch])

# shuffle finally and create a hashmap
for j in range(NUM_CLIENT):
    split_map[j] = idx_batch[j]
    test_split_map[j] = idxt_batch[j]
x_client_train={i: np.array([]) for i in range(NUM_CLIENT)}
y_client_train={i: np.array([]) for i in range(NUM_CLIENT)}
x_client_test={i: np.array([]) for i in range(NUM_CLIENT)}
y_client_test={i: np.array([]) for i in range(NUM_CLIENT)}
x_client_val={i: np.array([]) for i in range(NUM_CLIENT)}
y_client_val={i: np.array([]) for i in range(NUM_CLIENT)}

for i in range(NUM_CLIENT):
    for j in split_map[i]:
        x_client_train[i]=np.append(x_client_train[i],X_post[j])
        y_client_train[i]=np.append(y_client_train[i],y_post[j])
    x_client_train[i]=x_client_train[i].reshape(len(y_client_train[i]),3,32,32)

for i in range(NUM_CLIENT):
    for j in test_split_map[i]:
        x_client_test[i]=np.append(x_client_test[i],X_post_test[j])
        y_client_test[i]=np.append(y_client_test[i],y_post_test[j])
    x_client_test[i]=x_client_test[i].reshape(len(y_client_test[i]),3,32,32)
x_c_train={i: np.array([]) for i in range(NUM_CLIENT)}
y_c_train={i: np.array([]) for i in range(NUM_CLIENT)}
x_c_val={i: np.array([]) for i in range(NUM_CLIENT)}
y_c_val={i: np.array([]) for i in range(NUM_CLIENT)}

for i in range(NUM_CLIENT):
    x_c_train[i], x_c_val[i] = train_test_split(x_client_train[i], test_size=0.2,random_state=RANDOM_SEED)
    y_c_train[i], y_c_val[i] = train_test_split(y_client_train[i], test_size=0.2,random_state=RANDOM_SEED)
for i in range(NUM_CLIENT):
    x_client_train[i]=np.array(x_c_train[i])
    y_client_train[i]=np.array(y_c_train[i])
    x_client_val[i]=np.array(x_c_val[i])
    y_client_val[i]=np.array(y_c_val[i])
    x_client_test[i]=np.array(x_client_test[i])
    y_client_test[i]=np.array(y_client_test[i])

    x_client_train[i],y_client_train[i]=torch_tensor(x_client_train[i], y_client_train[i])
    x_client_val[i], y_client_val[i]=torch_tensor(x_client_val[i], y_client_val[i])
    x_client_test[i], y_client_test[i]=torch_tensor(x_client_test[i], y_client_test[i])
client_train={i: np.array([]) for i in range(NUM_CLIENT)}
client_val={i: np.array([]) for i in range(NUM_CLIENT)}
client_test={i: np.array([]) for i in range(NUM_CLIENT)}
#로컬 데이터 로드
for i in range(NUM_CLIENT):
    client_train[i] = DataLoader(
        TensorDataset(x_client_train[i], y_client_train[i]),
        batch_size=BATCH_SIZE_Po,
        shuffle=True,
    )
    client_val[i] = DataLoader(
        TensorDataset(x_client_val[i], y_client_val[i]),
        batch_size=BATCH_SIZE_Po,
        shuffle=True,
    )
    client_test[i] = DataLoader(
        TensorDataset(x_client_test[i], y_client_test[i]),
        batch_size=BATCH_SIZE_Po,
        shuffle=False,
    )
class FederatedData(torch.utils.data.Dataset):
    def __init__(self, x_dict, y_dict):
        self.x_dict = x_dict
        self.y_dict = y_dict
        self.idx_list = list(range(len(x_dict)))  # 인덱스 범위 설정

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        client_number = self.idx_list[idx]  # 인덱스 활용
        return {"data": self.x_dict[client_number], "target": self.y_dict[client_number]}
train_fed_data = FederatedData(x_client_train, y_client_train)
train_loader = DataLoader(train_fed_data, batch_size=BATCH_SIZE_Po, shuffle=False, num_workers=20, pin_memory=True)
# Pre_train_loader에서 전체 타겟 수집
all_targets = []
for _, target in Pre_train_loader:
    all_targets.append(target)

all_targets = torch.cat(all_targets)
dist = torch.bincount(all_targets, minlength=10).float() / len(all_targets)

# 기존 코드 스타일 유지
orig_dist = []
for _ in Pre_train_loader:  # 반복은 하지만 내부 계산은 동일 분포 사용
    orig_dist.append(dist)

# 모든 클라이언트에게 동일한 분포 할당
orig_dist = [dist.clone() for _ in range(NUM_CLIENT)]

# # orig_dist 시각화: 첫 5개 클라이언트 분포 보기
# plt.figure(figsize=(12, 4))
# for i in range(5):
#     plt.plot(orig_dist[i].cpu().numpy(), label=f"Client {i}")
# plt.title("Original (Pretrain) Class Distribution")
# plt.xlabel("Class")
# plt.ylabel("Proportion")
# plt.legend()
# plt.grid(True)
# plt.show()
# dist
new_dist = []
for idx, raw in enumerate(train_fed_data):
    # prepare data
    data = raw['data']
    target = raw['target']

    new_dist.append(target.bincount(minlength=10)/len(target))

# # orig_dist 시각화: 첫 5개 클라이언트 분포 보기
# plt.figure(figsize=(12, 4))
# for i in range(5):
#     plt.plot(new_dist[i].cpu().numpy(), label=f"Client {i}")
# plt.title("Original (Pretrain) Class Distribution")
# plt.xlabel("Class")
# plt.ylabel("Proportion")
# plt.legend()
# plt.grid(True)
# plt.show()

# Shift functions
def lin(t, T):
    return t / T

def squ(t, T):
    L = int(np.sqrt(T))
    return (t // (L // 2)) % 2

def sin(t, T):
    L = int(np.sqrt(T))
    return 0.5 * (1 + np.sin(2 * np.pi * (t % L) / L))

def ber(t, T, p=0.1):
    if t == 0:
        return 0
    if np.random.rand() < p:
        return 1 - ber(t - 1, T, p)
    else:
        return ber(t - 1, T, p)

shift_dist = [[] for _ in range(NUM_CLIENT)]  # client별 시간 분포 리스트

for client_id in range(NUM_CLIENT):
    for t in range(T):
        if shift == "lin":
            w = lin(t, T - 1)
        elif shift == "sin":
            w = sin(t, T - 1)
        elif shift == "squ":
            w = squ(t, T - 1)
        elif shift == "ber":
            w = ber(t, T - 1)
        else:
            print("shift 지정 에러")
        mix_dist = (1 - w) * orig_dist[client_id] + w * new_dist[client_id]
        shift_dist[client_id].append(mix_dist)

# # 클라이언트 0의 시간에 따른 클래스 분포 변화
# plt.figure(figsize=(12, 5))
# for cls in range(NUM_CLASSES):
#     class_probs = [shift_dist[0][t][cls].item() for t in range(T)]
#     plt.plot(class_probs, label=f"Class {cls}")
# plt.title("Client 0 - Class Distribution over Time (Linear Shift)")
# plt.xlabel("Time")
# plt.ylabel("Probability")
# plt.legend()
# plt.grid(True)
# plt.show()

# dist에 따른 데이터 샘플링 진행
label_index={i: [] for i in range(10)}

for idx, label in enumerate(y_post):
    label_index[label.item()].append(idx)

# 클라이언트마다 타임스텝별 데이터를 저장할 딕셔너리 초기화
x_client_adt = {i: {j: [] for j in range(T)} for i in range(NUM_CLIENT)}
y_client_adt = {i: {j: [] for j in range(T)} for i in range(NUM_CLIENT)}

# 각 클라이언트에 대해 시간에 따라 데이터 분포를 따라 샘플링
for i in range(NUM_CLIENT):
    for t in range(T):
        # 분포에 따라 레이블 샘플링
        p = shift_dist[i][t].cpu().numpy().astype(np.float64)
        p /= p.sum()
        labels = np.random.choice(NUM_CLASSES, sample_per_step, p=p)

        # 레이블에 맞는 실제 데이터 샘플링
        for label in labels:
            index = np.random.choice(label_index[label])  # 라벨에 맞는 데이터 인덱스 중 하나 선택
            x_client_adt[i][t].append(X_post[index])  # 이미지 추가
            y_client_adt[i][t].append(label)  # 라벨 추가

        # 리스트를 tensor로 변환
        x_client_adt[i][t] = torch.tensor(np.stack(x_client_adt[i][t]), dtype=torch.float32)
        y_client_adt[i][t] = torch.tensor(y_client_adt[i][t], dtype=torch.long)

# import matplotlib.pyplot as plt
# import numpy as np
#
# # 시각화할 클라이언트와 타임스텝 목록 지정
# client_to_plot = 0
# # time_steps_to_plot = [0, 25, 50, 75, 99]
# time_steps_to_plot = [0, 10,20,30,40,49]
#
# plt.figure(figsize=(15, 4))
#
# for i, t in enumerate(time_steps_to_plot):
#     plt.subplot(1, len(time_steps_to_plot), i + 1)
#
#     # 라벨 분포 계산
#     labels = y_client_adt[client_to_plot][t].cpu().numpy()
#     counts = np.bincount(labels, minlength=NUM_CLASSES)
#     probs = counts / sum(counts)
#
#     # 막대 그래프
#     plt.bar(range(NUM_CLASSES), probs)
#     plt.title(f"Client {client_to_plot} - Time {t}")
#     plt.ylim(0, 1)
#     plt.xlabel("Class")
#     if i == 0:
#         plt.ylabel("Proportion")
#     else:
#         plt.yticks([])
#
# plt.tight_layout()
# plt.show()

md
## 모델 정의
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MiniResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.5, lamda=lamda):
        super(MiniResNet, self).__init__()
        self.in_channels = 3

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2, dropout_rate=0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64))
        self.fc2 = nn.Sequential(
            nn.Linear(64, num_classes))
        self.lamda = lamda  # EWC 정규화 강도

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = F.leaky_relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

def MiniResNet6():
    return MiniResNet(BasicBlock, [1, 1, 1])
#로컬 디바이스 cnn 모델 생성
#모델도 여러개 생성
cnn_model=MiniResNet6().to(device)

optimizer=optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE_Pr)
loss_func=nn.NLLLoss().to(device)

md
## Pre train
# # 재현성을 위한 시드 설정
# torch.manual_seed(RANDOM_SEED)
#
# # 에포크별 손실 값을 저장할 리스트
# train_epoch_losses = []
# val_epoch_losses = []
#
# for epoch in tqdm(range(NUM_EPOCHS_Pr), desc="Centralized Training"):
#     # -------------------
#     # Training Phase
#     # -------------------
#     cnn_model.train()  # 학습 모드 전환
#     running_loss_train = 0.0
#     total_train_samples = 0
#
#     for data, target in Pre_train_loader:
#         data = data.to(device)
#         target = target.to(device, dtype=torch.long)
#
#         optimizer.zero_grad()
#         output = cnn_model(data)
#         log_probs = F.log_softmax(output, dim=1)
#         loss = loss_func(log_probs, target)
#         loss.backward()
#         optimizer.step()
#
#         # 배치별 손실 누적
#         running_loss_train += loss.item() * data.size(0)
#         total_train_samples += data.size(0)
#
#     epoch_loss_train = running_loss_train / total_train_samples
#     train_epoch_losses.append(epoch_loss_train)
# save_dir = "./model"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, f"pretrained_if{IF}_preset{num_pre_dataset}_seed{now_running}_CIFAR10.pth")
# torch.save(cnn_model.state_dict(), save_path)
## pretrained 모델 load
pre_model = MiniResNet6()
#save_path = f"./model/pretrained_uniform_CIFAR10.pth"
state_dict = torch.load(save_path,weights_only=True)
pre_model.load_state_dict(state_dict)
pre_model.eval()
#클라이언트 로컬 모델 정의
cnn_model_client={i: MiniResNet6().to(device) for i in range(NUM_CLIENT)}
optimizer_client={i: optim.Adam(cnn_model_client[i].parameters(), lr=LEARNING_RATE_Po) for i in range(NUM_CLIENT)}
loss_func_client={i: nn.NLLLoss().to(device) for i in range(NUM_CLIENT)}
## pretrained model deploy
for i in range(NUM_CLIENT):
    cnn_model_client[i].load_state_dict(pre_model.state_dict())

#중앙 서버 모델 생성 및 모델 파라미터 초기화
cnn_model_global=MiniResNet6().to(device)
cnn_model_global.load_state_dict(pre_model.state_dict())
# 공통: 예측 결과 미리 저장
def get_predictions_for_all_clients(model_dict, test_loader, num_clients, device):
    predictions = {}
    ground_truths = {}
    for client_id in range(num_clients):
        preds, targets = [], []
        model = model_dict[client_id]
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                if x.ndim == 3:
                    x = x.unsqueeze(1)
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out, 1)
                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())
        predictions[client_id] = np.array(preds)
        ground_truths[client_id] = np.array(targets)
    return predictions, ground_truths


# 함수 1: 특정 타임스텝 t에 대해 모든 클라이언트 POP 평균 성능 출력
def evaluate_pop_at_timestep(t, y_client_adt, model_dict, test_loader, num_clients, num_classes, device):
    # 예측 결과 미리 계산
    predictions, ground_truths = get_predictions_for_all_clients(model_dict, test_loader, num_clients, device)

    # 타임스텝 t의 POP 점수 저장
    pop_scores_t = []

    for cid in range(num_clients):
        # 타임스텝 t에서 클라이언트의 라벨 분포
        labels = y_client_adt[cid][t].cpu().numpy()
        label_dist = np.bincount(labels, minlength=num_classes).astype(np.float64)

        if label_dist.sum() == 0:
            pop_scores_t.append(0.0)
            continue

        weight = label_dist / label_dist.sum()

        # 예측 결과
        preds = predictions[cid]
        targets = ground_truths[cid]

        correct_per_class = np.zeros(num_classes)
        total_per_class = np.zeros(num_classes)

        for gt, pr in zip(targets, preds):
            if gt == pr:
                correct_per_class[gt] += 1
            total_per_class[gt] += 1

        # POP score 계산
        pop = sum(
            (correct_per_class[c] / total_per_class[c] if total_per_class[c] > 0 else 0.0) * weight[c]
            for c in range(num_classes)
        )
        pop_scores_t.append(pop * 100)  # %

    # 최종 출력
    mean_score = np.mean(pop_scores_t)
    std_score = np.std(pop_scores_t)

    # print(f"[Time {t}] 평균 POP: {mean_score:.2f}%, 클라이언트 간 표준편차: {std_score:.2f}")
    return mean_score, std_score, pop_scores_t  # <- pop_scores_t 추가 반환
    # return mean_score, std_score
def average_global_parameters(cnn_model_global, client_models, client_list, P_C):
    # 글로벌 파라미터 초기화
    global_state = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in cnn_model_global.state_dict().items()}

    # 클라이언트 파라미터 합산
    for i in client_list:
        client_state = client_models[i].state_dict()
        for name, param in client_state.items():
            global_state[name] += param.to(torch.float32)  # 실수형 변환

    # 클라이언트 수로 평균화
    for name in global_state:
        global_state[name] /= float(P_C)  # 실수형 나눗셈

    # 글로벌 모델 업데이트
    cnn_model_global.load_state_dict(global_state)

# ==== 하이브리드용 Helper들 (루프 위에 추가) ====

@torch.no_grad()
def get_embedding(model, x):
    was_training = model.training
    model.eval()
    out = model.layer1(x)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)       # (B, 128, 1, 1)
    out = torch.flatten(out, 1)    # (B, 128)
    model.train(was_training)
    return out
@contextmanager
def temp_model_on(device, state_dict):
    m = MiniResNet6().to(device)
    m.load_state_dict(state_dict, strict=True)
    try:
        yield m
    finally:
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@torch.no_grad()
def get_embedding_from_state(state_dict, x_cpu):
    with temp_model_on(device, state_dict) as model:
        model.eval()
        x = x_cpu.to(device, non_blocking=True)
        feats = model.layer1(x); feats = model.layer2(feats); feats = model.layer3(feats)
        feats = model.avgpool(feats)
        feats = torch.flatten(feats, 1)  # (B,128)
        return feats.detach().cpu()

@torch.no_grad()
def predict_from_state(state_dict, x_cpu):
    with temp_model_on(device, state_dict) as model:
        model.eval()
        x = x_cpu.to(device, non_blocking=True)
        logits = model(x)
        preds  = logits.softmax(dim=1).argmax(dim=1)
        return preds.detach().cpu()

def train_client_once(client_id, t, data_cpu, target_cpu, adaptive_lr, global_state_for_shared):
    """임시 모델을 올려 한 번 적응 학습하고 CPU state로 회수"""
    # 로딩 & shared 동기화
    local_sd = copy.deepcopy(client_state[client_id])
    # 글로벌 shared를 주입
    for k, v in global_state_for_shared.items():
        if k in local_sd:
            local_sd[k] = v.detach().cpu()

    with temp_model_on(device, local_sd) as model:
        model.train()
        x = data_cpu.to(device, non_blocking=True)
        y = target_cpu.to(device, non_blocking=True)
        opt = torch.optim.SGD(model.parameters(), lr=adaptive_lr)

        # (1) shared freeze 단계
        for n, p in model.named_parameters():
            p.requires_grad = not any(layer in n for layer in shared_layers)

        for _ in range(NUM_EPOCHS_FL_Po):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            probs  = logits.softmax(dim=1)
            loss   = -(probs * (probs.clamp_min(1e-10)).log()).sum(1).mean()
            loss.backward()
            opt.step()

        # (2) 전체 unfreeze 단계
        for p in model.parameters():
            p.requires_grad = True

        for _ in range(NUM_EPOCHS_FL_Po):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            probs  = logits.softmax(dim=1)
            loss   = -(probs * (probs.clamp_min(1e-10)).log()).sum(1).mean()
            loss.backward()
            opt.step()

        # 업데이트 회수 (CPU)
        client_state[client_id] = {k: v.detach().cpu() for k, v in model.state_dict().items()}


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """a,b: 1D 텐서. 반환: 파이썬 float"""
    a = a.float(); b = b.float()
    an = a.norm(p=2) + eps
    bn = b.norm(p=2) + eps
    return float(torch.dot(a, b) / (an * bn))
def average_global_parameters(cnn_model_global, client_models, client_list, P_C):
    # 글로벌 파라미터 초기화
    global_state = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in cnn_model_global.state_dict().items()}

    # 클라이언트 파라미터 합산
    for i in client_list:
        client_state = client_models[i].state_dict()
        for name, param in client_state.items():
            global_state[name] += param.to(torch.float32)  # 실수형 변환

    # 클라이언트 수로 평균화
    for name in global_state:
        global_state[name] /= float(P_C)  # 실수형 나눗셈

    # 글로벌 모델 업데이트
    cnn_model_global.load_state_dict(global_state)
# 각 클라이언트별 이전 상태 버퍼
prev_pred_dist = {}      # {cid: Tensor(NUM_CLASSES)}
prev_embed_mean = {}     # {cid: Tensor(128)}


# %% [cell 1]
start_time = time.time()
torch.manual_seed(RANDOM_SEED)
prev_dist  = {} 
eval_loss_arr = []
eval_acc_arr = []
# 루프 시작 전
prev_feat_mean = {}     # {client_id: torch.Tensor(D)}
feat_mean_ema  = {}     # {client_id: torch.Tensor(D)}  # 선택: EMA로 기준을
selected_lr_dict = {}  # (t, round, client) -> lr

for t in range(T):
    for round in range(ROUNDS):
        random.seed(random_list[round])
        client_list = np.sort(random.sample(n_c, p_c))

        # --- shared layers update (글로벌 → 로컬 동기화) ---
        for i in range(NUM_CLIENT):
            cnn_model_client[i].train()
            global_state_dict = cnn_model_global.state_dict()
            local_state_dict  = cnn_model_client[i].state_dict()
            for layer_name in shared_layers:
                for param_type in ['weight', 'bias']:
                    key = f"{layer_name}.{param_type}"
                    if key in global_state_dict:
                        local_state_dict[key] = global_state_dict[key]
            cnn_model_client[i].load_state_dict(local_state_dict)

        # --- personal layers update ---
        for i in client_list:
            data = x_client_adt[i][t].to(device)
            target = y_client_adt[i][t].to(device).long()
            if data.shape[1] != 1:
                data = data

            # ===== (A) Label-shift 신호: (1 - cosine similarity) =====
            with torch.no_grad():
                model_output = cnn_model_client[i](data)
                preds = F.softmax(model_output, dim=1).argmax(dim=1)

            num_classes = NUM_CLASSES
            current_dist = torch.bincount(preds, minlength=num_classes).float()
            current_dist = current_dist / (current_dist.sum() + 1e-12)

            if (i not in prev_dist) or (t == 0):
                prev_dist[i] = current_dist.clone()

            cos_sim = F.cosine_similarity(
                prev_dist[i].unsqueeze(0), current_dist.unsqueeze(0)
            ).item()
            S_label = 1.0 - cos_sim             # ↑ 변화 크면 커짐, [이론상 0~2]

            # ===== (B) Covariate-shift 신호: mean(embedding) Cosine =====
            with torch.no_grad():
                feats = get_embedding(cnn_model_client[i], data)  # (B, D)
            
            # 현재 배치 임베딩 평균 (L2 정규화해서 방향 유사도만 보게 함)
            cur_mean = F.normalize(feats.mean(dim=0), dim=0)
            
            # 기준(mean) 없으면 초기화
            if i not in prev_feat_mean:
                prev_feat_mean[i] = cur_mean.detach().clone()
                feat_mean_ema[i]  = cur_mean.detach().clone()   # 선택: EMA 기준도 같이 시작
                cos_sim_feat = 1.0                               # 동일하다고 보고 shift=0
            else:
                # 이전 EMA-기준과의 코사인 유사도
                ref_mean = F.normalize(feat_mean_ema[i], dim=0)
                cos_sim_feat = F.cosine_similarity(
                    ref_mean.unsqueeze(0), cur_mean.unsqueeze(0), dim=1
                ).item()
            
            # 변화량(shift score): [-1,1]의 cos → [0,1]로 매핑
            S_cov = max(0.0, min(1.0, 0.5 * (1.0 - cos_sim_feat)))
            
            # EMA로 기준 업데이트(더 매끄럽게)
            beta_ema = 0.9
            feat_mean_ema[i] = F.normalize(
                beta_ema * feat_mean_ema[i] + (1.0 - beta_ema) * cur_mean.detach(),
                dim=0
            )
            prev_feat_mean[i] = feat_mean_ema[i].detach().clone()

            # ===== (C) 하이브리드 게이트 + 적응 LR =====
            
            G =0.5*S_label +  0.5*S_cov
            adaptive_lr = LR_MIN + G * (LR_MAX - LR_MIN)
            
            optimizer = torch.optim.SGD(cnn_model_client[i].parameters(), lr=adaptive_lr)

            # ===== 두 단계 로컬 업데이트(원래 코드 유지) =====
            # (1) shared freeze
            for name, para in cnn_model_client[i].named_parameters():
                if any(layer in name for layer in shared_layers):
                    para.requires_grad = False
                else:
                    para.requires_grad = True

            for _ in range(NUM_EPOCHS_FL_Po):
                optimizer.zero_grad()
                logits = cnn_model_client[i](data)
                probs = F.softmax(logits, dim=1)
                entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                entropy_loss.backward()
                optimizer.step()

            # (2) unfreeze all
            for name, para in cnn_model_client[i].named_parameters():
                para.requires_grad = True

            for _ in range(NUM_EPOCHS_FL_Po):
                optimizer.zero_grad()
                logits = cnn_model_client[i](data)
                probs = F.softmax(logits, dim=1)
                entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                entropy_loss.backward()
                optimizer.step()

            selected_lr_dict[(t, round, i)] = adaptive_lr
            prev_dist[i] = current_dist.clone()   # ← 네 원래 코드와 동일 타이밍으로 갱신
            del (data, target)

        # --- global averaging (네가 쓰던 방식 그대로 유지) ---
        global_state_dict = cnn_model_global.state_dict()

        for key in global_state_dict:
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])

        for i in client_list:
            client_state_dict = cnn_model_client[i].state_dict()
            for layer_name in shared_layers:
                for key in global_state_dict:
                    if layer_name in key:
                        global_state_dict[key] += client_state_dict[key]

        for layer_name in shared_layers:
            for key in global_state_dict:
                if layer_name in key:
                    global_state_dict[key] = global_state_dict[key] / len(client_list)

        cnn_model_global.load_state_dict(global_state_dict)

    # ===== 시점 t에서 즉시 평가 (학습에 쓴 데이터로 CE/Acc) =====
    total_loss, total_correct, total_num = 0.0, 0, 0
    for i in range(NUM_CLIENT):
        cnn_model_client[i].eval()
        data = x_client_adt[i][t].to(device)
        target = y_client_adt[i][t].to(device).long()
        if data.shape[1] != 1:
            data = data
        with torch.no_grad():
            logits = cnn_model_client[i](data)
            probs = F.softmax(logits, dim=1)
            loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
            pred = probs.argmax(dim=1)
            correct = (pred == target).sum().item()
            total_loss += loss * data.size(0)
            total_correct += correct
            total_num += data.size(0)
    avg_loss = total_loss / max(1, total_num)
    avg_acc  = total_correct / max(1, total_num)
    eval_loss_arr.append(avg_loss)
    eval_acc_arr.append(avg_acc)
    #print(f"[t={t}] Train-data Eval Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
final_acc=np.mean(eval_acc_arr)*100

print("\n전체 타임스텝 평균 LAcc: {:.2f}%".format(np.mean(eval_acc_arr)*100))

