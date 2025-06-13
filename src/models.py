import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    T = cfg.model.temporal_window if "temporal_window" in cfg.model else 1
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["temporal_window"] = T
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    model_kwargs_clean = {k: v for k, v in model_kwargs.items() if k != "temporal_window"}
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "unet_se":
        model = UNetWithSEBlock(**model_kwargs)
    elif cfg.model.type == "unet_cbam":
        model = UNetWithCBAM(**model_kwargs)
    elif cfg.model.type == "unet_vit":
        model = UNetWithViTBottleneck(**model_kwargs_clean)
    elif cfg.model.type == "convlstm_unet":
        model = ConvLSTMUNet(**model_kwargs)
    elif cfg.model.type == "convlstm_unet_se":
        model = ConvLSTMUNetSE(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


# --- Model Architectures ---

# ---------------- given base model ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
        temporal_window=None
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x

# ---------------- model 2 [Unet + SE] ----------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            SEBlock(out_c)
        )

    def forward(self, x):
        return self.block(x)

class UNetWithSEBlock(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_c=64, dropout_rate=0.2, temporal_window=3):
        super().__init__()
        self.enc1 = ConvBlock(n_input_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_c * 2, base_c * 4)
        self.dropout = nn.Dropout2d(dropout_rate)
    
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(base_c * 4, base_c * 2),
            nn.Dropout2d(dropout_rate)
        )

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(base_c * 2, base_c),
            nn.Dropout2d(dropout_rate)
        )
    
        # self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, stride=2)
        # self.dec2 = ConvBlock(base_c * 4, base_c * 2)
        # self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, 2, stride=2)
        # self.dec1 = ConvBlock(base_c * 2, base_c)

        self.final = nn.Conv2d(base_c, n_output_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        b = self.dropout(b)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


## ---------------- model 3  [Unet + CBAM] ----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        return self.sigmoid(avg_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class UNetWithCBAM(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_c=64, dropout_rate=0.2, temporal_window=1):
        super().__init__()
        in_channels = n_input_channels * temporal_window

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_c)
        self.cbam1 = CBAM(base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.cbam2 = CBAM(base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 2, base_c * 4)
        self.cbam_bottleneck = CBAM(base_c * 4)
        self.dropout = nn.Dropout2d(dropout_rate)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)
        self.cbam_dec2 = CBAM(base_c * 2)

        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)
        self.cbam_dec1 = CBAM(base_c)

        # Final
        self.final = nn.Conv2d(base_c, n_output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.cbam1(self.enc1(x))                # [B, base_c, H, W]
        e2 = self.cbam2(self.enc2(self.pool1(e1)))   # [B, base_c*2, H/2, W/2]

        # Bottleneck
        b = self.cbam_bottleneck(self.bottleneck(self.pool2(e2)))
        b = self.dropout(b)

        # Decoder
        up2 = self.up2(b)
        d2 = self.cbam_dec2(self.dec2(torch.cat([up2, e2], dim=1)))

        up1 = self.up1(d2)
        d1 = self.cbam_dec1(self.dec1(torch.cat([up1, e1], dim=1)))

        return self.final(d1)


## ---------------- model 4  [Unet + VIT] ----------------
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetWithViTBottleneck(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_c=64, dropout_rate=0.2, img_size=(48, 72), temporal_window=3):
        super().__init__()

        self.encoder1 = ConvBlock(n_input_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck: ViT
        patch_h, patch_w = 6, 6
        from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
        self.flatten_hw = (img_size[0] // 4, img_size[1] // 4)
        vit_dim = base_c * 4
        num_patches = (self.flatten_hw[0] // patch_h) * (self.flatten_hw[1] // patch_w)

        self.bottleneck_conv = nn.Conv2d(base_c * 2, vit_dim, kernel_size=1)
        self.vit = TransformerEncoder(
            TransformerEncoderLayer(d_model=vit_dim, nhead=4, dim_feedforward=vit_dim * 2),
            num_layers=2
        )
        self.unbottleneck_conv = nn.Conv2d(vit_dim, base_c * 2, kernel_size=1)

        # Decoder
        self.up2      = nn.ConvTranspose2d(base_c * 2, base_c * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(base_c * 4, base_c * 2)   # 256-in → 128-out
        
        self.up1      = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(base_c * 2, base_c)


        self.final_tas = nn.Conv2d(base_c, 1, kernel_size=1)  # temperature
        self.final_pr = nn.Conv2d(base_c, 1, kernel_size=1)   # precipitation

        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.pool2(e2)

        # ViT bottleneck
        b = self.bottleneck_conv(b)  # (B, D, H, W)
        B, D, H, W = b.shape
        b_flat = b.flatten(2).permute(2, 0, 1)  # (HW, B, D)
        b_vit = self.vit(b_flat).permute(1, 2, 0).reshape(B, D, H, W)
        b = self.unbottleneck_conv(b_vit)

        d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = self.dropout(d1)                     # (B, 64, 48, 72)
    
        out_tas = self.final_tas(d1)              # use d1, not x
        out_pr  = self.final_pr(d1)               # use d1, not x
        return torch.cat([out_tas, out_pr], dim=1)  # (B, 2, 48, 72)

## ---------------- model 5  [Unet + ConvLSTM] ----------------
class Temporal3DEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x3d):  # [B, T, C, H, W]
        x3d = x3d.permute(0, 2, 1, 3, 4)  # → [B, C, T, H, W]
        out = self.conv3d(x3d)           # → [B, out_ch, T, H, W]
        out_last = out[:, :, -1, :, :]   # Collapse T: pick last frame
        return out_last                  # → [B, out_ch, H, W]

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)  # [B, C+H, H, W]
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x_seq):  # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.size()
        h, c = (torch.zeros(B, self.cell.conv.out_channels // 4, H, W, device=x_seq.device) for _ in range(2))
        for t in range(T):
            h, c = self.cell(x_seq[:, t], h, c)
        return h  # return last hidden state


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ConvLSTMUNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_c=64, temporal_window=None):
        super().__init__()
        
        self.base_c = base_c 
        self.encoder1 = DecoderBlock(n_input_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DecoderBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.convlstm = ConvLSTM(base_c * 2, base_c * 4)
        self.temporal3d = Temporal3DEncoder(in_ch=n_input_channels, out_ch=base_c)

        
        # ==== TAS decoder branch ====
        self.up2_tas  = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2_tas = DecoderBlock(base_c * 4, base_c * 2)
        self.up1_tas  = nn.ConvTranspose2d(base_c * 2, base_c,     kernel_size=2, stride=2)
        self.dec1_tas = DecoderBlock(base_c * 2, base_c)
        self.final_tas = nn.Conv2d(base_c, 1, kernel_size=1)  # produces 1 channel (tas)

        # ==== PR decoder branch ====
        self.up2_pr   = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2_pr  = DecoderBlock(base_c * 4, base_c * 2)
        self.up1_pr   = nn.ConvTranspose2d(base_c * 2, base_c,     kernel_size=2, stride=2)
        self.dec1_pr  = DecoderBlock(base_c * 2, base_c)
        self.final_pr = nn.Conv2d(base_c, 1, kernel_size=1)  # produces 1 channel (pr)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        bc = self.base_c  # alias for convenience

        # --- TEMPORAL 3D ENCODER ---
        t3d_out = self.temporal3d(x)  # [B, bc, H, W]
        
        # --- 1) Encoder 1 over time frames ---
        x_seq = x.view(B * T, C, H, W)
        e1    = self.encoder1(x_seq).view(B, T, bc, H, W)
        e1_t  = e1[:, -1, :, :, :] + t3d_out  # fused features

        # # --- 1) Encoder 1 over time frames ---
        # x_seq = x.view(B * T, C, H, W)          # [B*T, C,  H,   W]
        # e1    = self.encoder1(x_seq)            # [B*T, bc, H,   W]
        # e1    = e1.view(B, T, bc, H, W)         # [B, T, bc, H, W]
        # e1_t  = e1[:, -1, :, :, :]              # [B, bc, H, W]
    
        # --- 2) Encoder 2 over pooled features ---
        p1    = e1.reshape(B * T, bc, H, W)     # [B*T, bc, H,   W]
        p1    = self.pool1(p1)                  # [B*T, bc, H/2, W/2]
        e2    = self.encoder2(p1)               # [B*T, bc*2, H/2, W/2]
        e2    = e2.view(B, T, bc * 2, H // 2, W // 2)  # [B, T, 2·bc, H/2, W/2]
        e2_t  = e2[:, -1, :, :, :]              # [B, 2·bc, H/2, W/2]
    
        # --- 3) ConvLSTM bottleneck ---
        p2    = e2.reshape(B * T, bc * 2, H // 2, W // 2)  # [B*T, 2·bc, H/2, W/2]
        p2    = self.pool2(p2)                             # [B*T, 2·bc, H/4, W/4]
        x_seq_lstm = p2.view(B, T, bc * 2, H // 4, W // 4)  # [B, T, 2·bc, H/4, W/4]
        b = self.convlstm(x_seq_lstm)                       # [B, 4·bc, H/4, W/4]

        
        # --- 4) Tas decoder branch ---
        up2_tas_out = self.up2_tas(b)                        # [B, 2·bc, H/2, W/2]
        d2_tas       = self.dec2_tas(torch.cat([up2_tas_out, e2_t], dim=1))  # [B, 2·bc, H/2, W/2]
        up1_tas_out  = self.up1_tas(d2_tas)                   # [B, bc, H, W]
        d1_tas       = self.dec1_tas(torch.cat([up1_tas_out, e1_t], dim=1))   # [B, bc, H, W]
        out_tas      = self.final_tas(d1_tas)                 # [B, 1, H, W]
    
        # --- 5) Pr decoder branch ---
        up2_pr_out = self.up2_pr(b)                           # [B, 2·bc, H/2, W/2]
        d2_pr       = self.dec2_pr(torch.cat([up2_pr_out, e2_t], dim=1))       # [B, 2·bc, H/2, W/2]
        up1_pr_out  = self.up1_pr(d2_pr)                      # [B, bc, H, W]
        d1_pr       = self.dec1_pr(torch.cat([up1_pr_out, e1_t], dim=1))       # [B, bc, H, W]
        out_pr      = self.final_pr(d1_pr)                    # [B, 1, H, W]
    
        return torch.cat([out_tas, out_pr], dim=1)             # [B, 2, H, W]

########################## ablation studies ##########################

## ---------------- model 6  [Unet + ConvLSTM No temporal encoder] ----------------
# class ConvLSTMUNet(nn.Module):
#     def __init__(self, n_input_channels, n_output_channels, base_c=64, temporal_window=None):
#         super().__init__()

#         self.base_c = base_c
#         self.encoder1 = DecoderBlock(n_input_channels, base_c)
#         self.pool1 = nn.MaxPool2d(2)

#         self.encoder2 = DecoderBlock(base_c, base_c * 2)
#         self.pool2 = nn.MaxPool2d(2)

#         self.convlstm = ConvLSTM(base_c * 2, base_c * 4)

#         # ==== TAS decoder branch ====
#         self.up2_tas = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
#         self.dec2_tas = DecoderBlock(base_c * 4, base_c * 2)
#         self.up1_tas = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
#         self.dec1_tas = DecoderBlock(base_c * 2, base_c)
#         self.final_tas = nn.Conv2d(base_c, 1, kernel_size=1)

#         # ==== PR decoder branch ====
#         self.up2_pr = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
#         self.dec2_pr = DecoderBlock(base_c * 4, base_c * 2)
#         self.up1_pr = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
#         self.dec1_pr = DecoderBlock(base_c * 2, base_c)
#         self.final_pr = nn.Conv2d(base_c, 1, kernel_size=1)

#     def forward(self, x):  # x: (B, T, C, H, W)
#         B, T, C, H, W = x.shape
#         bc = self.base_c

#         # --- Encoder 1 ---
#         x_seq = x.view(B * T, C, H, W)
#         e1 = self.encoder1(x_seq).view(B, T, bc, H, W)
#         e1_t = e1[:, -1, :, :, :]  # Removed t3d_out fusion

#         # --- Encoder 2 ---
#         p1 = e1.view(B * T, bc, H, W)
#         p1 = self.pool1(p1)
#         e2 = self.encoder2(p1).view(B, T, bc * 2, H // 2, W // 2)
#         e2_t = e2[:, -1, :, :, :]

#         # --- ConvLSTM bottleneck ---
#         p2 = e2.view(B * T, bc * 2, H // 2, W // 2)
#         p2 = self.pool2(p2)
#         x_seq_lstm = p2.view(B, T, bc * 2, H // 4, W // 4)
#         b = self.convlstm(x_seq_lstm)

#         # --- Tas decoder ---
#         up2_tas_out = self.up2_tas(b)
#         d2_tas = self.dec2_tas(torch.cat([up2_tas_out, e2_t], dim=1))
#         up1_tas_out = self.up1_tas(d2_tas)
#         d1_tas = self.dec1_tas(torch.cat([up1_tas_out, e1_t], dim=1))
#         out_tas = self.final_tas(d1_tas)

#         # --- Pr decoder ---
#         up2_pr_out = self.up2_pr(b)
#         d2_pr = self.dec2_pr(torch.cat([up2_pr_out, e2_t], dim=1))
#         up1_pr_out = self.up1_pr(d2_pr)
#         d1_pr = self.dec1_pr(torch.cat([up1_pr_out, e1_t], dim=1))
#         out_pr = self.final_pr(d1_pr)

#         return torch.cat([out_tas, out_pr], dim=1)


## ---------------- model 7  [Unet + ConvLSTM No separate decoder head] -----------
# class ConvLSTMUNet(nn.Module):
#     def __init__(self, n_input_channels, n_output_channels=2, base_c=64, temporal_window=None):
#         super().__init__()

#         self.base_c = base_c 
#         self.encoder1 = DecoderBlock(n_input_channels, base_c)
#         self.pool1 = nn.MaxPool2d(2)

#         self.encoder2 = DecoderBlock(base_c, base_c * 2)
#         self.pool2 = nn.MaxPool2d(2)

#         self.convlstm = ConvLSTM(base_c * 2, base_c * 4)
#         self.temporal3d = Temporal3DEncoder(in_ch=n_input_channels, out_ch=base_c)

#         # Shared decoder
#         self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
#         self.dec2 = DecoderBlock(base_c * 4, base_c * 2)
#         self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
#         self.dec1 = DecoderBlock(base_c * 2, base_c)
#         self.final = nn.Conv2d(base_c, n_output_channels, kernel_size=1)  # shared output

#     def forward(self, x):  # x: (B, T, C, H, W)
#         B, T, C, H, W = x.shape
#         bc = self.base_c

#         # Temporal encoding
#         t3d_out = self.temporal3d(x)

#         # Encoder path
#         x_seq = x.view(B * T, C, H, W)
#         e1 = self.encoder1(x_seq).view(B, T, bc, H, W)
#         e1_t = e1[:, -1, :, :, :] + t3d_out

#         p1 = e1.view(B * T, bc, H, W)
#         p1 = self.pool1(p1)
#         e2 = self.encoder2(p1).view(B, T, bc * 2, H // 2, W // 2)
#         e2_t = e2[:, -1, :, :, :]

#         p2 = e2.view(B * T, bc * 2, H // 2, W // 2)
#         p2 = self.pool2(p2)
#         x_seq_lstm = p2.view(B, T, bc * 2, H // 4, W // 4)
#         b = self.convlstm(x_seq_lstm)

#         # Shared decoder
#         up2_out = self.up2(b)
#         d2 = self.dec2(torch.cat([up2_out, e2_t], dim=1))
#         up1_out = self.up1(d2)
#         d1 = self.dec1(torch.cat([up1_out, e1_t], dim=1))
#         out = self.final(d1)  # [B, 2, H, W]

#         return out
