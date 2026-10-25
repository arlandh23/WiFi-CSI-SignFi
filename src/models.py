import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d_block(cin, cout, k, pool=True, drop=0.1):
    layers = [nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool1d(2))
    layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers)


class Baseline1DCNN(nn.Module):
    """Baseline: subcarrier x kanal dijadikan channel, konvolusi di sumbu waktu."""
    def __init__(self, in_ch=180, n_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            conv1d_block(in_ch, 128, 7), conv1d_block(128, 128, 5),
            conv1d_block(128, 256, 3), conv1d_block(256, 256, 3, pool=False))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):                      # x: (B, C, S, T)
        x = x.flatten(1, 2)                    # (B, C*S, T)
        return self.head(self.features(x))


class CNNBiLSTMAttention(nn.Module):
    """Proposed: CNN (fitur lokal) -> BiLSTM (temporal) -> temporal attention pooling."""
    def __init__(self, in_ch=180, n_classes=7, conv_ch=128, hidden=128, layers=2):
        super().__init__()
        self.cnn = nn.Sequential(conv1d_block(in_ch, conv_ch, 5), conv1d_block(conv_ch, conv_ch, 5))
        self.lstm = nn.LSTM(conv_ch, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(0.3), nn.Linear(2 * hidden, n_classes))
        self.last_attention = None

    def forward(self, x):                      # (B, C, S, T)
        h = self.cnn(x.flatten(1, 2))          # (B, conv_ch, T/4)
        h, _ = self.lstm(h.transpose(1, 2))    # (B, T/4, 2H)
        w = torch.softmax(self.attn(h).squeeze(-1).float(), dim=1)   # (B, T/4)
        self.last_attention = w.detach()
        z = (h * w.unsqueeze(-1).to(h.dtype)).sum(1)
        return self.head(z)


class ResBlock2d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential() if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class SpecResNet(nn.Module):
    """ResNet kecil untuk spektrogram PCA-STFT (B, 2*N_PCA, F, T)."""
    def __init__(self, in_ch=8, n_classes=7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layers = nn.Sequential(ResBlock2d(32, 32), ResBlock2d(32, 64, 2), ResBlock2d(64, 128, 2), nn.Dropout2d(0.1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
