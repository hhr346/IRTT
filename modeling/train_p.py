import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Tools import irtt_config
from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import time
import random


# ---------------------------
# Model / training hyperparams (tweak as needed)
# ---------------------------
n_features = 154
n_layer = 4
n_head = 8
n_embd = 256         # embedding dimension
batch_size = 128
lr = 5e-5
dropout = 0.1
weight_decay = 1e-2

num_epochs = 300    # Number of training epochs
device = torch.device("cuda:0")
params = irtt_config('../params.json')
suffix = "_final_p_300"

save_name = f'./test/{params.gas_name}_training_IRTT{suffix}.png'
model_path = f"./model_path/{params.gas_name}_IRTT_model{suffix}.pth"
normalizer_path = f'./model_path/{params.gas_name}_IRTT_normalizer{suffix}.npz'
print(f"n_feature: {n_features}, n_embd: {n_embd}, Batch size: {batch_size}, Device: {device}, Save name: {save_name}")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ---------------------------
# Data prep helpers
# ---------------------------
def normalize(data, min, max, scale=1, log=False):
    data = data * scale
    if log:
        data = np.log(data)
    data = (data - min) / (max - min)
    data = data * 2 - 1
    return data


def prepare_data(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.15, test_ratio: float = 0.1, shuffle: bool = True):
    assert X.ndim == 2 and X.shape[1] == n_features
    assert y.ndim == 1 and y.shape[0] == X.shape[0]
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    test_n = int(N * test_ratio)
    val_n = int(N * val_ratio)
    test_idx = idx[:test_n]
    val_idx = idx[test_n:test_n + val_n]
    train_idx = idx[test_n + val_n:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def fit_channel_normalizer(X_train: np.ndarray, eps=1e-8):
    # per-channel mean/std (over samples)
    mean = X_train.mean(axis=0, keepdims=True)  # (1, features)
    std = X_train.std(axis=0, keepdims=True)
    std[std < eps] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean) / std).astype(np.float32)

# Batch generator (samples random indices each call)
def get_batch(X_arr, y_arr, batch_size):
    N = X_arr.shape[0]
    ix = np.random.randint(0, N, size=(batch_size,))
    xb = torch.from_numpy(X_arr[ix]).to(device)          # (B, T)
    yb = torch.from_numpy(y_arr[ix]).to(device)          # (B,)
    return xb, yb

def forward_in_batches(model, X, batch_size=64):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            mu, log_var = model(xb)
            preds_list.append(mu.cpu().numpy())
    return np.concatenate(preds_list, axis=0)

# ---------------------------
# Model: Encoder-only Transformer for regression
# ---------------------------
class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)    # (B, T, hs)
        v = self.value(x)  # (B, T, hs)
        # scaled dot-product attention (no causal mask)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concat heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EncoderRegressor(nn.Module):
    def __init__(self, seq_len, n_features, n_embd=128, n_layer=4, n_head=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        # project raw per-channel value into embedding
        self.scalar_proj = nn.Linear(1, n_embd)

        # BUT here we want to treat channels as "sequence" positions: reshape below
        # positional embeddings for sequence positions (channels)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, n_embd))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # regression head: reduce sequence dimension via mean pooling then MLP -> scalar
        self.reg_head = nn.Sequential(
            nn.Linear(n_embd, n_embd//2),
            nn.ReLU(),
            nn.Linear(n_embd//2, 2)
        )

    def forward(self, x):
        # x: (B, T) where T == seq_len (each channel is a token value)
        B, T = x.shape
        assert T == self.seq_len

        # use linear applied to last dim after unsqueeze
        # project each scalar to embedding
        x_in = x.unsqueeze(-1)             # (B, T, 1)
        x_emb = self.scalar_proj(x_in)         # (B, T, n_embd)

        # add positional embeddings
        x_emb = x_emb + self.pos_emb[:, :T, :].to(x_emb.device)
        x_emb = self.dropout(x_emb)
        x = self.blocks(x_emb)  # (B, T, n_embd)
        x = self.ln_f(x)
        # global average pool over sequence (channels)
        x_pooled = x.mean(dim=1)  # (B, n_embd)
        out = self.reg_head(x_pooled).squeeze(-1)  # (B,)
        mu = out[:, 0]
        log_var = out[:, 1]
        return mu, log_var


# ---------------------------
# Metrics
# ---------------------------
def mse(a, b):
    return float(((a - b) ** 2).mean())
def rmse(a, b):
    return float(np.sqrt(mse(a, b)))

def nll_loss(mu, log_var, y):
    # log_var is predicted log(σ²)
    return torch.mean(0.5 * (torch.exp(-log_var) * (y - mu)**2 + log_var))

def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

# ---------------------------
# Training routine
# ---------------------------
def train_model(X: np.ndarray, y: np.ndarray):
    # split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(X, y, val_ratio=0.15, test_ratio=0.1, shuffle=True)

    # normalizer on training set (per-channel)
    mean, std = fit_channel_normalizer(X_train)
    # Save the normalizer
    np.savez(normalizer_path, mean=mean, std=std, label_min=label_min, label_max=label_max)
    # apply normalizer
    X_train_n = apply_normalizer(X_train, mean, std)
    X_val_n   = apply_normalizer(X_val, mean, std)
    X_test_n  = apply_normalizer(X_test, mean, std)

    print("Train/Val/Test sizes:", X_train_n.shape[0], X_val_n.shape[0], X_test_n.shape[0])
    model = EncoderRegressor(seq_len=n_features, n_features=n_features,
                             n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    global_step = 0
    train_loss_list = []
    val_loss_list = []

    model.train()
    for epoch in range(num_epochs):
        t0 = time.time()
        # iterate approximate number of batches per epoch
        steps_per_epoch = max(1, X_train_n.shape[0] // batch_size)
        for it in range(steps_per_epoch):
            xb, yb = get_batch(X_train_n, y_train, batch_size)
            # start_idx = it * batch_size
            # end_idx = start_idx + batch_size
            # xb, yb = torch.from_numpy(X[start_idx:end_idx]).to(device), torch.from_numpy(y[start_idx:end_idx]).to(device)

            # xb: (B, T) numpy->tensor already
            mu, log_var = model(xb)  # (B,)
            loss = nll_loss(mu, log_var, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

        # Full val loss at epoch end
        model.eval()
        losses_train = []
        losses = []
        with torch.no_grad():
            train_preds = forward_in_batches(model, X_train_n, batch_size=batch_size)
            preds = forward_in_batches(model, X_test_n, batch_size=batch_size)
            losses_train.append(criterion(torch.from_numpy(train_preds).to(device), torch.from_numpy(y_train).to(device)).item())
            losses.append(criterion(torch.from_numpy(preds).to(device), torch.from_numpy(y_test).to(device)).item())

        train_loss = float(np.mean(losses_train))
        val_loss = float(np.mean(losses))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch+1:03d} Step {global_step:06d} TrainLoss {train_loss:.6f} ValLoss {val_loss:.6f}")
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs} done in {epoch_time:.1f}s")


    # Save the model
    torch.save(model.state_dict(), model_path)
    # For scatter/hist2d plotting collect train predictions across epochs (for final figure)
    all_train_labels = []
    all_train_outputs = []
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        # train set
        train_preds = forward_in_batches(model, X_train_n, batch_size=batch_size)
        all_train_labels.append(y_train)
        all_train_outputs.append(train_preds)
        # test set
        preds = forward_in_batches(model, X_test_n, batch_size=batch_size)
        all_labels.append(y_test)
        all_outputs.append(preds)

    all_outputs = np.concatenate(all_outputs, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0).squeeze()
    all_train_outputs = np.concatenate(all_train_outputs, axis=0).squeeze()
    all_train_labels = np.concatenate(all_train_labels, axis=0).squeeze()

    r2 = r2_score(all_labels, all_outputs)
    mape = mean_absolute_percentage_error(all_labels, all_outputs)
    mse = mean_squared_error(all_labels, all_outputs)
    rmse = np.sqrt(mse)

    r2_train = r2_score(all_train_labels, all_train_outputs)
    mape_train = mean_absolute_percentage_error(all_train_labels, all_train_outputs)
    mse_train = mean_squared_error(all_train_labels, all_train_outputs)
    rmse_train = np.sqrt(mse_train)

    print(f"Validation R2: {r2:.5f}")
    print(f"Validation MAPE: {mape:.5f}")
    print(f"Validation RMSE: {rmse:.5f}")



    # 绘制学习曲线
    plt.figure(figsize=(20, 6), dpi=300)
    # 学习曲线子图
    plt.subplot(1, 3, 1)
    plt.scatter(range(num_epochs), train_loss_list, s=10)
    plt.plot(train_loss_list, label='Train Loss')
    plt.scatter(range(num_epochs), val_loss_list, s=10)
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
 
    # 训练集散点分布图子图
    plt.subplot(1, 3, 2)
    x_edges = np.linspace(-1, 1, 201)   # 比如横轴从 0 到 1，分成 10 个区间
    y_edges = np.linspace(-1, 1, 201)  # 纵轴从 -1 到 1，分成 20 个区间

    h = plt.hist2d(
        all_train_labels, all_train_outputs,
        bins=[x_edges, y_edges],  # 手动指定 bin 边界
        cmap='jet',
        norm=LogNorm()
    )
    # plt.colorbar(h[3], label='Counts')
    plt.colorbar(h[3])

    plt.plot([min(all_train_labels), max(all_train_labels)], [min(all_train_labels), max(all_train_labels)], 
            'r--', label='Ideal Line')

    # 添加拟合直线 (线性回归线)
    fit_coef = np.polyfit(all_train_labels, all_train_outputs, 1)
    fit_line = np.poly1d(fit_coef)
    if fit_coef[1] < 0:
        symbol = ''
    else:
        symbol = '+'
    plt.plot(all_train_labels, fit_line(all_train_labels), color='gray', linestyle='--', 
             label=f'Fit Line (y={fit_coef[0]:.4f}x{symbol}{fit_coef[1]:.4f})')

    # 添加指标文本
    textstr = '\n'.join((
        f'$\\mathrm{{R}}^2$ = {r2_train:.4f}',
        f'MSE = {mse_train:.4f}',
        f'RMSE = {rmse_train:.4f}'))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend(loc='lower right')
    # plt.title('Training Set: True vs Predicted Values', fontsize=20)
    plt.grid(True, alpha=0.3)

    # 测试集预测分布图
    plt.subplot(1, 3, 3)
    # plt.scatter(all_labels, all_outputs, alpha=0.5, s=10)

    h = plt.hist2d(
        all_labels, all_outputs,
        bins=[x_edges, y_edges],  # 手动指定 bin 边界
        cmap='jet',
        norm=LogNorm()
    )
    # plt.colorbar(h[3], label='Counts')
    plt.colorbar(h[3])

    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 
            'r--', label='Ideal Line')

    # 添加拟合直线 (线性回归线)
    fit_coef = np.polyfit(all_labels, all_outputs, 1)
    fit_line = np.poly1d(fit_coef)
    if fit_coef[1] < 0:
        symbol = ''
    else:
        symbol = '+'
    plt.plot(all_labels, fit_line(all_labels), color='gray', linestyle='--', 
            label=f'Fit Line (y={fit_coef[0]:.4f}x{symbol}{fit_coef[1]:.4f})')

    # 添加指标文本
    textstr = '\n'.join((
        f'$\\mathrm{{R}}^2$ = {r2:.4f}',
        f'MSE = {mse_train:.4f}',
        f'RMSE = {rmse:.4f}'))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()



if __name__ == '__main__':
    OUTPUT_OUT = params.output_folder + f"{params.gas_name}/dataset_simulate/uniform/"
    filename = 'total_dataset_full'
    filepath = f'{OUTPUT_OUT}/{filename}.nc'

    ncfile = nc.Dataset(filepath, mode='r')
    print(f"Opening file {filepath}")
    inputs = np.asarray(ncfile.variables['input'][:, :])[:, :]
    inputs = np.concatenate([inputs[:, :153], inputs[:, 154:155]], axis=1)

    labels = np.asarray(ncfile.variables['output'][:])[:]
    lat = np.asarray(ncfile.variables['lat'][:])
    lon = np.asarray(ncfile.variables['lon'][:])
    ncfile.close()

    index = (inputs[:, 2] > -1)
    inputs = inputs[index, :]
    labels = labels[index]

    label_min, label_max = np.min(labels)/1E18, np.max(labels)/1E18
    print(f"The min and max of labels before normalization are {label_min}, {label_max}")
    labels = normalize(labels, label_min, label_max, 1E-18)
    print(f"The shape of inputs is {inputs.shape}, the shape of labels is {labels.shape}")
    print(f"The labels variance is {np.var(labels)}")
    train_model(inputs, labels)
