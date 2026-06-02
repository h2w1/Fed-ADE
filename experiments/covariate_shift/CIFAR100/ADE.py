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
# ======================================================================
# CIFAR-100-C Federated Post-Training (OOM-safe)
# - Load pretrained MiniResNet (best_miniresnet_cifar100.pth)
# - Clients keep only CPU state_dict (no GPU residency)
# - Per-client training uses a temporary GPU model
# - Global averaging only for shared float tensors (CPU float32), then load to GPU
# - Covariate shift = cosine( prev_embed_mean , curr_embed_mean )  -> S_cov = (1 - cos)/2
# - Label shift     = cosine( prev_pred_dist , curr_pred_dist )    -> S_label = 1 - cos
# ======================================================================

import os, gc, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# -------------------------
# 기본 설정
# -------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if CUDA_VISIBLE_DEVICES_OVERRIDE is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES_OVERRIDE)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gc.collect()
if device == 'cuda':
    torch.cuda.empty_cache()
#print("[Device]", device)

# -------------------------
# 하이퍼파라미터 / 설정
# -------------------------
#now_running = 0
random_seed_list = [1752, 42, 153, 248, 300, 439, 517, 694, 752, 846]
RANDOM_SEED = random_seed_list[now_running]
np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#np.random.seed(RANDOM_SEED)
#random_list = np.random.randint(low=1, high=1000, size=ROUNDS)

NUM_CLASSES = 100
NUM_CLIENT = globals().get("NUM_CLIENT", 100)
CLIENT_FRAC = 0.1
P_C         = int(NUM_CLIENT * CLIENT_FRAC)

T = globals().get("T", 100)     # 총 타임스텝
ROUNDS = globals().get("ROUNDS", 50)      # 타임스텝 내 로컬 라운드
sample_per_step = globals().get("sample_per_step", 30)
shared_layers   = ['layer1', 'layer2', 'layer3']

# 로컬 업데이트 스텝 (freeze / unfreeze 각각 반복 수)
NUM_EPOCHS_FL_Po = 3

# Adaptive learning-rate gate
LR_MAX = globals().get("LR_MAX", 1e-4)
LR_MIN = globals().get("LR_MIN", 1e-5)

#shift  = 'ber'      # 'lin'|'sin'|'squ'|'ber'

C=0.1
CR = np.arange(ROUNDS)
n_c = list(range(NUM_CLIENT))
p_c = int(len(n_c) * C)
np.random.seed(RANDOM_SEED)
random_list = np.random.randint(low=1, high=1000, size=ROUNDS)
# -------------------------
# CIFAR-100 정규화 상수
# -------------------------
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD  = [0.2675, 0.2565, 0.2761]
MEAN_T = torch.tensor(CIFAR100_MEAN).view(1,3,1,1)
STD_T  = torch.tensor(CIFAR100_STD).view(1,3,1,1)

def normalize_inplace_cifar100(x: torch.Tensor):
    """x: (B,3,H,W) float; 0..255 또는 0..1 -> CIFAR-100 통계로 정규화"""
    if x.max() > 1.0:
        x.div_(255.0)
    x.sub_(MEAN_T.to(x.device)).div_(STD_T.to(x.device))
    return x

# -------------------------
# 모델 (MiniResNet for 100 classes)
# -------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out

class MiniResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, dropout_rate=0.5, lamda: float = 0.0):
        super().__init__()
        self.in_channels = 3
        self.layer1 = self._make_layer(block, 32,  num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 64,  num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2, dropout_rate=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(nn.Linear(128, 64))     # Sequential 유지
        self.fc2 = nn.Sequential(nn.Linear(64, num_classes))
        self.lamda = lamda

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)     # (B,32,32,32)
        x = self.layer2(x)     # (B,64,16,16)
        x = self.layer3(x)     # (B,128,8,8)
        x = self.avgpool(x)    # (B,128,1,1)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x

def MiniResNet6(num_classes=100, dropout_rate=0.5, lamda: float = 0.0):
    return MiniResNet(BasicBlock, [1,1,1], num_classes=num_classes, dropout_rate=dropout_rate, lamda=lamda)

# -------------------------
# 프리트레인 로드
# -------------------------
PRETRAIN_PATH = globals().get("pretrained_path", "./model/pretrained_uniform_CIFAR100.pth")

def load_pretrained(model, path):
    if not os.path.isfile(path):
        #print(f"[WARN] Pretrained not found at {path}. Using randomly initialized weights.")
        return model
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    #print(f"[OK] Loaded pretrained from {path}")
    return model

pre_model = MiniResNet6().to('cpu')
pre_model = load_pretrained(pre_model, PRETRAIN_PATH)
pre_model.eval()

# -------------------------
# CIFAR-100-C 로드 & 시뮬레이션 배치 구성
# -------------------------
CIFAR100C_DIR = globals().get("cifar_c_dir", "./CIFAR-100-C")
assert os.path.isdir(CIFAR100C_DIR), "CIFAR-100-C 폴더가 필요합니다."

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
labels_path = os.path.join(CIFAR100C_DIR, "labels.npy")
c_labels = np.load(labels_path).astype(np.int64)  # 50k

corr_arrays = {}
for name in CORRUPTIONS:
    arr = np.load(os.path.join(CIFAR100C_DIR, f"{name}.npy")).astype("float32")  # (50000,32,32,3)
    corr_arrays[name] = arr

def get_severity_slice(sev:int):
    start = (sev - 1) * 10000
    end   = sev * 10000
    return slice(start, end)

def lin(t,T): return t/T
def sinf(t,T):
    L = int(np.sqrt(T)); return 0.5*(1+np.sin(2*np.pi*(t % L)/L))
def squ(t,T):
    L = int(np.sqrt(T)); return (t // (L//2)) % 2
def ber(t,T,p=0.1):
    if t == 0: return 0
    if np.random.rand() < p: return 1 - ber(t-1,T,p)
    return ber(t-1,T,p)

def severity_at_t(t, T, mode='ber'):
    if mode=='lin': w = lin(t, T-1)
    elif mode=='sin': w = sinf(t, T-1)
    elif mode=='squ': w = squ(t, T-1)
    elif mode=='ber': w = ber(t, T-1)
    else: raise ValueError("shift mode error")
    return int(np.clip(np.ceil(w*4)+1, 1, 5))

rng = np.random.RandomState(RANDOM_SEED)
client_corr_type = {i: CORRUPTIONS[i % len(CORRUPTIONS)] for i in range(NUM_CLIENT)}

x_client_adt = {i: {t: None for t in range(T)} for i in range(NUM_CLIENT)}
y_client_adt = {i: {t: None for t in range(T)} for i in range(NUM_CLIENT)}

for i in range(NUM_CLIENT):
    corr_name = client_corr_type[i]
    arr = corr_arrays[corr_name]  # (50000,32,32,3)
    for t in range(T):
        sev = severity_at_t(t, T, shift)  # 1..5
        sl = get_severity_slice(sev)
        idx = rng.choice(np.arange(sl.start, sl.stop), size=sample_per_step, replace=False)
        imgs = arr[idx]                               # (B,32,32,3) 0..255 float32
        labs = c_labels[idx % 10000]                  # (B,)
        xt = torch.tensor(np.transpose(imgs,(0,3,1,2)), dtype=torch.float32)  # (B,3,32,32)
        normalize_inplace_cifar100(xt)                # 정규화
        x_client_adt[i][t] = xt                       # CPU tensor
        y_client_adt[i][t] = torch.tensor(labs, dtype=torch.long)

# -------------------------
# (핵심) 클라이언트 상태는 CPU에만 보관
# -------------------------
client_state = {i: copy.deepcopy(pre_model.state_dict()) for i in range(NUM_CLIENT)}

# 글로벌 모델은 GPU 상주
cnn_model_global = MiniResNet6().to(device)
cnn_model_global.load_state_dict(pre_model.state_dict())
client_state = {i: copy.deepcopy(pre_model.state_dict()) for i in range(NUM_CLIENT)}

from torch import nn, optim
cnn_model_client={i: MiniResNet6().to(device) for i in range(NUM_CLIENT)}
#optimizer_client={i: optim.Adam(cnn_model_client[i].parameters(), lr=LEARNING_RATE_Po) for i in range(NUM_CLIENT)}
#loss_func_client={i: nn.NLLLoss().to(device) for i in range(NUM_CLIENT)}
## pretrained model deploy
for i in range(NUM_CLIENT):
    cnn_model_client[i].load_state_dict(pre_model.state_dict())
# -------------------------
# @torch.no_grad()
# def get_embedding(model, x):
#     was_training = model.training
#     model.eval()
#     feats = model(x.to(device), feature=True)  # (B, 512) for ResNet18
#     model.train(was_training)
#     return feats


# %% [cell 1]
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

# -------------------------
# 게이트 보조 함수들
# -------------------------
def lambda_label(t: int, T: int, floor: float = 0.1) -> float:
    """초반 라벨 신호↑, 후반 코바리엇 신호↑. floor는 하한."""
    val = 1.0 - (t / max(1, (T - 1)))
    return float(max(floor, val))

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
            #lam = lambda_label(t, T, floor=beta)   # beta=1.0이면 lam=1 고정 → S_label만 사용
            G = 0.5*( S_label +  S_cov)
            adaptive_lr = LR_MIN + G * (LR_MAX - LR_MIN)
            #adaptive_lr = LR_MIN + S_cov * (LR_MAX - LR_MIN)
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

