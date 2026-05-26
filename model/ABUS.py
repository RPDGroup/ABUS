import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from model.ViT_pytorch import Encoder
from losses.proto_loss import ProtoLoss
from losses.supCon_loss import SupConLoss

class WindowAttention(nn.Module):
    """Window Attention"""
    def __init__(self, hidden_size, window_size, n_heads, qkv_bias=True, attn_dropout=0., dropout=0., use_relative_position_bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.n_heads = n_heads
        head_dim = hidden_size // n_heads
        self.scale = head_dim ** -0.5
        self.use_relative_position_bias = use_relative_position_bias
        
        if self.use_relative_position_bias:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_heads))
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        else:
            self.relative_position_bias_table = None
            self.relative_position_index = None
            
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: [batch_size * num_windows, window_size * window_size, hidden_size]
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        # qkv: [3, batch_size * num_windows, n_heads, window_size * window_size, hidden_size // n_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # Window Attention
        # attn: [batch_size * num_windows, n_heads, window_size * window_size, window_size * window_size]
        
        if self.use_relative_position_bias:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
            
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.n_heads, N, N) + mask.unsqueeze(1)
            # [batch_size, num_windows, n_heads, window_size * window_size, window_size * window_size]
            attn = attn.view(-1, self.n_heads, N, N)
            # [batch_size * num_windows, n_heads, window_size * window_size, window_size * window_size]
        
        # attn map
        attn_map = F.softmax(attn, dim=-1)
        attn_drop = self.attn_drop(attn_map)
        
        x = (attn_drop @ v).transpose(1, 2).reshape(B_, N, C)
        # [batch_size * num_windows, window_size * window_size, hidden_size]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_map
        


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, hidden_size, input_resolution, n_heads, window_size=4, shift_size=0, mlp_ratio=4, dropout=0., attn_dropout=0., use_relative_position_bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_resolution = input_resolution # [img_size // patch_size * img_size // patch_size]
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = WindowAttention(
            hidden_size=hidden_size, window_size=(self.window_size, self.window_size), n_heads=n_heads, 
            attn_dropout=attn_dropout, dropout=dropout, use_relative_position_bias=use_relative_position_bias
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self._window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        
    def _window_partition(self, x, window_size):
        """Window Partition"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        # num_windows = H // window_size * W // window_size
        return windows # [batch_size * num_windows, window_size, window_size, hidden_size]
    
    def _window_reverse(self, windows, window_size, H, W):
        """Window Reverse"""
        # windows: [batch_size * num_windows, window_size, window_size, hidden_size]
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        # [batch_size, img_size // patch_size // window_size, img_size // patch_size // window_size, window_size, window_size, hidden_size]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x # [batch_size, img_size // patch_size, img_size // patch_size, hidden_size]
        
    def forward(self, x):
        # x: [batch_size, img_size // patch_size * img_size // patch_size, embedding_dim]
        H, W = self.input_resolution
        B, L, C = x.shape
        x_residual = x
        x = self.norm_1(x)
        x = x.view(B, H, W, C) # [batch_size, img_size // patch_size, img_size // patch_size, hidden_size]
        
        # Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        x_windows = self._window_partition(shifted_x, self.window_size) # [batch_size * num_windows, window_size, window_size, hidden_size]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # [batch_size * num_windows, window_size * window_size, hidden_size]
        
        # Window Attention
        x_windows, attn = self.attn(x_windows, mask=self.attn_mask)
        
        # [batch_size * num_windows, window_size * window_size, hidden_size]
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        # [batch_size * num_windows, window_size, window_size, hidden_size]
        shifted_x = self._window_reverse(x_windows, self.window_size, H, W)
        
        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C) # [batch_size, img_size // patch_size * img_size // patch_size, hidden_size]
        x = x + x_residual
        x = x + self.mlp(self.norm2(x))
        
        # x: [batch_size, img_size // patch_size * img_size // patch_size, hidden_size]
        # attn_map: [batch_size * num_windows, n_heads, window_size * window_size, window_size * window_size]
        return x, attn
        
        
class PatchEmbedding(nn.Module):
    """Patch Embedding"""
    def __init__(self, img_size=32, patch_size=4, in_channels=13, embedding_dim=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        # x: [batch_size, num_channels, img_size, img_size]
        x = self.conv(x) # [batch_size, embedding_dim, img_size // patch_size, img_size // patch_size]
        x = x.flatten(2) # [batch_size, embedding_dim, img_size // patch_size * img_size // patch_size]
        x = x.transpose(1, 2) # [batch_size, img_size // patch_size * img_size // patch_size, embedding_dim]
        x = self.norm(x)
        return x
        

class SwinBranch(nn.Module):
    """Swin Branch"""
    def __init__(self, config, img_size=32, channels=13):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        patch_size = config.patch_size
        window_size = config.window_size
        depths = config.swin_depths
        num_heads = config.swin_num_heads
        dropout = config.drop_path_rate
        use_relative_position_bias = config.use_relative_position_bias
        shift_size = config.shift_size
        
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=channels, embedding_dim=self.hidden_size)
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        
        self.blocks = nn.ModuleList()
        for i in range(sum(depths)):
            stage_idx = [i < s for s in itertools.accumulate(depths)].index(True)
            n_heads = num_heads[stage_idx]
            if shift_size != 0:
                shift = 0 if (i % 2 == 0) else shift_size // 2
            else:
                shift = 0
            
            self.blocks.append(
                SwinTransformerBlock(
                    hidden_size=self.hidden_size,
                    input_resolution=patches_resolution,
                    n_heads=n_heads,
                    window_size=window_size,
                    shift_size=shift,
                    dropout=dropout,
                    attn_dropout=dropout,
                    use_relative_position_bias=use_relative_position_bias
                )
            )
        self.norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, x):
        # x: [batch_size, num_channels, img_size, img_size]
        x = self.patch_embedding(x) # [batch_size, img_size // patch_size * img_size // patch_size, embedding_dim]
        
        all_attentions = []
        for block in self.blocks:
            x, attn = block(x)
            # x: [batch_size, img_size // patch_size * img_size // patch_size, hidden_size]
            # attn_map: [batch_size * num_windows, n_heads, window_size * window_size, window_size * window_size]
            all_attentions.append(attn)
        x = self.norm(x)
        x = x.mean(dim=1) # [batch_size, 1, hidden_size]
        
        return x, all_attentions
        
        


class ABUS(nn.Module):
    """ABUS model"""
    def __init__(self, config, img_size=32, margin=0, temperature=0.5, get_embedding=False):
        super().__init__()
        
        self.index_dict = {
            'shape_complementarity': (0, 5, 1, 6, 10),
            'RASA': (11, 12, 10),
            'hydrogen_bonds': (2, 7, 10),
            'charge': (3, 8, 10),
            'hydrophobicity': (4, 9, 10)
        }
        
        self.img_size = img_size
        self.get_embedding = get_embedding
        self.spatial_swin_transformer_list = nn.ModuleList()
        for feature in self.index_dict.keys():
            self.spatial_swin_transformer_list.append(
                self._init_swin_transformer(config, channels=len(self.index_dict[feature]))
            )
        self.feature_transformer = Encoder(config, vis=True)
            
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size), requires_grad=True) # [1, 1, hidden_size]
        self.proto_vector_pos = nn.Parameter(torch.rand(1, config.hidden_size), requires_grad=True) # [1, hidden_size]
        self.proto_vector_neg = nn.Parameter(torch.rand(1, config.hidden_size), requires_grad=True) # [1, hidden_size]
        
        self.margin = margin
        self.temperature = temperature
        self.proto_loss_fn = ProtoLoss(margin=self.margin)
        self.bce_loss_fn = nn.CrossEntropyLoss()
        self.SupCon_loss_fn = SupConLoss(temperature=self.temperature, base_temperature=self.temperature*10)
        
            
    def _init_swin_transformer(self, config, channels):
        return SwinBranch(config, img_size=self.img_size, channels=channels)
    
    def forward(self, img, labels=None):
        # img: [batch_size, num_channels, img_size, img_size]
        all_x = []
        all_spatial_attn = []
        
        # Swin Branch
        for i, feature in enumerate(self.index_dict.keys()):
            img_channels = img[:, self.index_dict[feature], :, :]
            x, attn = self.spatial_swin_transformer_list[i](img_channels)
            # x: [batch_size, 1, hidden_size]
            # attn_map: [batch_size * num_windows, n_heads, window_size * window_size, window_size * window_size] * 5 * num_blocks
            all_x.append(x)
            all_spatial_attn.append(attn)
            
        x = torch.stack(all_x, dim=1) # [batch_size, 5, hidden_size]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [batch_size, 1, hidden_size]
        x = torch.cat((cls_tokens , x), dim=1) # [batch_size, 6, hidden_size]
        
        x, feature_attn = self.feature_transformer(x)
        # x: [batch_size, 6, hidden_size]
        
        x = x[:, 0]
        x = F.normalize(x)
        
        if self.get_embedding:
            return x
        
        proto_vector_pos = F.normalize(self.proto_vector_pos)
        proto_vector_neg = F.normalize(self.proto_vector_neg)
        
        dist = nn.PairwiseDistance()
        dist_to_pos_prototype = dist(x, proto_vector_pos.repeat(x.shape[0], 1))
        dist_to_neg_prototype = dist(x, proto_vector_neg.repeat(x.shape[0], 1))
        
        logits = torch.stack([dist_to_pos_prototype - dist_to_neg_prototype,
                            dist_to_neg_prototype - dist_to_pos_prototype], axis=1)
        # [batch_size, 2]
        scores = dist_to_pos_prototype - dist_to_neg_prototype
        
        if labels is not None:
            proto_loss = self.proto_loss_fn(x, proto_vector_pos, proto_vector_neg, labels)
            BCE_loss = self.bce_loss_fn(logits, labels)
            supCon_loss = self.SupCon_loss_fn(x, labels)
            loss = proto_loss + BCE_loss + supCon_loss
            return scores, (all_spatial_attn, feature_attn), loss
        else:
            return scores, (all_spatial_attn, feature_attn)
        
            