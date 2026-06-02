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
# =========================
# TinyImageNet + Hybrid LR (Label + Covariate W2)
# =========================
import os, time, gc, random
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
from contextlib import contextmanager
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset


# %% [cell 1]
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # ← 두 GPU 노출

#dist="zipf"
#now_running = 1
LR_MAX = globals().get("LR_MAX", 1e-4)
LR_MIN = globals().get("LR_MIN", 5e-6)
LEARNING_RATE_Po = 5e-4
#shift = 'ber'  # lin | sin | squ | ber
save_path = f"./model/pretrained_{dist}_TinyImageNet.pth"
#save_path = f"./model/pretrained_TinyImageNet.pth"
pretrained_path = save_path


# %% [cell 2]
# -------------------------
# Device & Determinism
# -------------------------
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -------------------------
# Config
# -------------------------

RANDOM_SEED_LIST = [42, 153, 248, 300, 439, 517, 694, 752, 846]
RANDOM_SEED = RANDOM_SEED_LIST[now_running]
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fed settings
NUM_CLASSES   = 200
NUM_CLIENT = globals().get("NUM_CLIENT", 100)
C             = 0.1
P_C           = int(NUM_CLIENT * C)
ROUNDS = globals().get("ROUNDS", 10)
T = globals().get("T", 100)
sample_per_step = globals().get("sample_per_step", 100)
shared_layers = ['conv1','layer1','layer2','layer3','layer4']

# LR gate

# shift schedule on labels

beta   = 0.1          # <- 여기 값이 'lambda의 하한'. beta=1.0이면 라벨 시프트만 반영(기존과 동일)

# local steps
NUM_EPOCHS_FL_Po =  2
BATCH_SIZE_Po    = 256

# DP 옵션 (메모리 비용 큼! 기본 False 권장)
USE_DATAPARALLEL = False



# 랜덤 스케줄러
n_c = list(range(NUM_CLIENT))
p_c = int(len(n_c) * C)
random_list = np.random.randint(low=1, high=1000, size=ROUNDS)

# -------------------------
# ResNet (TinyImageNet용)
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
            )
    def forward(self, x):
        return F.relu(self.residual_function(x) + self.shortcut(x), inplace=True)

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BottleNeck.expansion, 1, bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
    def forward(self, x):
        return F.relu(self.residual_function(x) + self.shortcut(x), inplace=True)

class ResNet(nn.Module):
    def __init__(self, dataset, block, num_block):
        super().__init__()
        if dataset == 'tiny':
            num_classes = 200
        elif dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'imagenet':
            num_classes = 1000
        else:
            raise ValueError('Incorrect Dataset Input.')

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(block, 64,  num_block[0], 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        out = self.conv1(x)
        out = self.layer1(out); out = self.layer2(out)
        out = self.layer3(out); out = self.layer4(out)
        out = self.avg_pool(out); out = torch.flatten(out, 1)
        if feature:
            return out
        return self.fc(out)

def resnet18(dataset): return ResNet(dataset, BasicBlock, [2,2,2,2])

# -------------------------
# Data Load / Cache
# -------------------------
def save_or_load_tensor(data_path, load_fn, *args, **kwargs):
    if os.path.exists(data_path):
        #print(f"Loading cached data from {data_path}")
        return torch.load(data_path)
    else:
        #print(f"Processing and saving to {data_path}")
        data = load_fn(*args, **kwargs)
        torch.save(data, data_path)
        return data

def load_tiny_imagenet_train(train_dir=globals().get("tiny_imagenet_train_dir", "./tiny-imagenet-200/train")):
    class_names = sorted(os.listdir(train_dir))
    class_map = {name: idx for idx, name in enumerate(class_names)}
    X, y = [], []
    for class_name in class_names:
        img_dir = os.path.join(train_dir, class_name, "images")
        for img_file in os.listdir(img_dir):
            path = os.path.join(img_dir, img_file)
            img = cv2.imread(path)
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            X.append(img); y.append(class_map[class_name])
    X = np.transpose(np.array(X), (0,3,1,2))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long), class_map

def load_tiny_imagenet_val_sampled(val_dir=globals().get("tiny_imagenet_val_dir", "./tiny-imagenet-200/val"), class_map=None, samples_per_class=30):
    val_annotations = os.path.join(val_dir, "val_annotations.txt")
    image_dir = os.path.join(val_dir, "images")
    df = {}
    with open(val_annotations, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            df[parts[0]] = parts[1]

    if class_map is None:
        class_names = sorted(os.listdir(globals().get("tiny_imagenet_train_dir", "./tiny-imagenet-200/train")))
        class_map = {name: idx for idx, name in enumerate(class_names)}

    class_to_images = defaultdict(list)
    for img_name, label in df.items():
        class_to_images[label].append(img_name)

    X, y = [], []
    for class_name, img_names in class_to_images.items():
        selected_imgs = img_names[:samples_per_class]
        for img_name in selected_imgs:
            path = os.path.join(image_dir, img_name)
            img = cv2.imread(path)
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            X.append(img); y.append(class_map[class_name])

    X = np.transpose(np.array(X), (0,3,1,2))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

train_cache_path = "tiny_imagenet_train.pt"
val_cache_path   = "tiny_imagenet_val.pt"
X_train, y_train, class_map = save_or_load_tensor(train_cache_path, load_tiny_imagenet_train)
X_test,  y_test  = save_or_load_tensor(val_cache_path,   load_tiny_imagenet_val_sampled, class_map=class_map, samples_per_class=30)

# Split
from sklearn.model_selection import train_test_split
X_train_np, y_train_np = X_train.numpy(), y_train.numpy()
X_test_np,  y_test_np  = X_test.numpy(),  y_test.numpy()

X_pre, X_post, y_pre, y_post = train_test_split(X_train_np, y_train_np, test_size=1/5, random_state=RANDOM_SEED, stratify=y_train_np)
X_t,   X_post_test, y_t, y_post_test = train_test_split(X_test_np, y_test_np, test_size=0.5, random_state=RANDOM_SEED, stratify=y_test_np)

X_pre  = torch.tensor(X_pre,  dtype=torch.float32); y_pre  = torch.tensor(y_pre,  dtype=torch.long)
X_post = torch.tensor(X_post, dtype=torch.float32); y_post = torch.tensor(y_post, dtype=torch.long)
X_t    = torch.tensor(X_t,    dtype=torch.float32)
X_post_test = torch.tensor(X_post_test, dtype=torch.float32)
y_t    = torch.tensor(y_t,    dtype=torch.long)
y_post_test = torch.tensor(y_post_test, dtype=torch.long)

# -------------------------
# Label shift 스케줄 생성
# -------------------------
def lin(t,T): return t / T
def squ(t,T):
    L = int(np.sqrt(T))
    return (t // (L // 2)) % 2
def sin(t,T):
    L = int(np.sqrt(T))
    return 0.5 * (1 + np.sin(2*np.pi * (t % L) / L))
def ber(t, T, p=0.1):
    if t == 0: return 0
    return 1 - ber(t-1, T, p) if np.random.rand() < p else ber(t-1, T, p)

# original / client별 분포
def torch_tensor(x, y):
    if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32)
    else: x = x.clone().detach()
    if isinstance(y, np.ndarray): y = torch.tensor(y, dtype=torch.long)
    else: y = y.clone().detach()
    return x, y

# 원래 분포(orig)와 클라별 분포(new)를 만들고, 시점 t에 따라 혼합
# Pre_train 분포(라벨 기준)
all_targets = y_pre  # 충분히 큼
dist = torch.bincount(all_targets, minlength=NUM_CLASSES).float()
dist = dist / dist.sum()

orig_dist = [dist.clone() for _ in range(NUM_CLIENT)]

# 각 클라이언트가 학습 후(test split)에서 갖는 분포(라벨 기반)
new_dist = []
for cid in range(NUM_CLIENT):
    # 간단히: y_post_test에서 해당 클라이언트에 대응하는 실제 분포를 만든다
    # 여기서는 균일하게 y_post_test 전체 분포 사용(원본 코드와 유사 동작)
    nd = torch.bincount(y_post_test, minlength=NUM_CLASSES).float()
    new_dist.append(nd / nd.sum())

# t별 label 샘플링을 위한 인덱스
label_index = {i: torch.where(y_post == i)[0].numpy() for i in range(NUM_CLASSES)}

x_client_adt = {i: {} for i in range(NUM_CLIENT)}
y_client_adt = {i: {} for i in range(NUM_CLIENT)}

for cid in range(NUM_CLIENT):
    for t in range(T):
        if shift == "lin":   w = lin(t, T-1)
        elif shift == "sin": w = sin(t, T-1)
        elif shift == "squ": w = squ(t, T-1)
        elif shift == "ber": w = ber(t, T-1)
        else: raise ValueError("shift 지정 에러")
        mix = (1-w)*orig_dist[cid] + w*new_dist[cid]
        p = (mix / mix.sum()).cpu().numpy()

        labels = np.random.choice(NUM_CLASSES, size=sample_per_step, p=p)
        idxs   = np.array([np.random.choice(label_index[l]) for l in labels])

        xb = X_post[idxs].float()
        yb = torch.tensor(labels, dtype=torch.long)
        x_client_adt[cid][t] = xb
        y_client_adt[cid][t] = yb

# -------------------------
# 모델들 준비 (pretrained 불러오기)
# -------------------------
pre_model = resnet18('tiny').to(device)
#save_path = f"./model/pretrained_if1_preset80000_seed0_TinyImageNet.pth"
if os.path.isfile(save_path):
    state_dict = torch.load(save_path, map_location='cpu',weights_only=True)
    pre_model.load_state_dict(state_dict, strict=False)
pre_model.eval()

if USE_DATAPARALLEL and torch.cuda.device_count() > 1:
    pre_model = nn.DataParallel(pre_model)

cnn_model_global = resnet18('tiny').to(device)
cnn_model_global.load_state_dict(pre_model.module.state_dict() if isinstance(pre_model, nn.DataParallel) else pre_model.state_dict())
cnn_model_client = {i: resnet18('tiny').to(device) for i in range(NUM_CLIENT)}
for i in range(NUM_CLIENT):
    cnn_model_client[i].load_state_dict(cnn_model_global.state_dict())

# -------------------------
# 하이브리드 게이팅 보조 함수
# -------------------------
@torch.no_grad()
def get_embedding(model, x):
    was_training = model.training
    model.eval()
    feats = model(x.to(device), feature=True)  # (B, 512) for ResNet18
    model.train(was_training)
    return feats


# %% [cell 3]
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


# %% [cell 4]
start_time = time.time()
torch.manual_seed(RANDOM_SEED)
prev_dist  = {} 
eval_loss_arr = []
eval_acc_arr = []

# ➕ 라운드별(타임스텝 t) 검증 기록 리스트
round_val_losses = []
round_val_accs   = []

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
            
            # 현재 배치 임베딩 평균 (L2 정규화)
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
            
            # EMA로 기준 업데이트
            beta_ema = 0.9
            feat_mean_ema[i] = F.normalize(
                beta_ema * feat_mean_ema[i] + (1.0 - beta_ema) * cur_mean.detach(),
                dim=0
            )
            prev_feat_mean[i] = feat_mean_ema[i].detach().clone()

            G = 0.5 * (S_label + S_cov)
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
            prev_dist[i] = current_dist.clone()
            del (data, target)

        # --- global averaging (원 방식 유지) ---
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

    # ➕ 라운드별(=타임스텝 t) 기록
    round_val_losses.append(float(avg_loss))
    round_val_accs.append(float(avg_acc))

    eval_loss_arr.append(avg_loss)
    eval_acc_arr.append(avg_acc)
    #print(f"[t={t}] Train-data Eval Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

end_time = time.time()
elapsed_time = time.time() - start_time
final_acc = np.mean(eval_acc_arr) * 100.0

print("\n전체 타임스텝 평균 LAcc: {:.2f}%".format(np.mean(eval_acc_arr)*100))

# ===== 결과 파일 저장 (.npy / .json) =====
import json
from pathlib import Path

out_dir = Path("./results")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / f"SEAD_{shift}_losse.npy", np.array(round_val_losses, dtype=np.float32))
np.save(out_dir / f"SEAD_{shift}_acc.npy",   np.array(round_val_accs,   dtype=np.float32))

with open(out_dir / f"SEAD_{shift}_losse.json", "w", encoding="utf-8") as f:
    json.dump([float(x) for x in round_val_losses], f, ensure_ascii=False, indent=2)
with open(out_dir / f"SEAD_{shift}_acc.json", "w", encoding="utf-8") as f:
    json.dump([float(x) for x in round_val_accs],   f, ensure_ascii=False, indent=2)


# %% [cell 5]
