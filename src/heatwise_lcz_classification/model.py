import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class LESA(nn.Module):
    """Local-enhanced self-attention for 2D feature tokens."""

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head

        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.proj = nn.Linear(inner, dim)

    def forward(self, x, return_attn=False, hw=None):
        B, N, C = x.shape
        if hw is None:
            H = W = int(N ** 0.5)
        else:
            H, W = hw
            assert H * W == N, "H*W must match token number N"

        x2d = x.transpose(1, 2).reshape(B, C, H, W)
        xloc = self.dw(x2d)
        xloc = xloc.flatten(2).transpose(1, 2)

        qkv = self.to_qkv(xloc).chunk(3, dim=-1)
        q, k, v = (t.view(B, N, self.heads, -1).transpose(1, 2) for t in qkv)

        attn = (q @ k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.proj(out)

        if return_attn:
            return attn
        return out


class CrossModalInteraction(nn.Module):
    """Feature-level interaction between HSI and MSI/Sentinel-2 branches."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.hsi_to_msi_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        self.msi_to_hsi_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, hsi_feat, msi_feat):
        hsi_attention = self.hsi_to_msi_attention(hsi_feat)
        msi_attention = self.msi_to_hsi_attention(msi_feat)

        hsi_enhanced = hsi_feat * msi_attention + hsi_feat
        msi_enhanced = msi_feat * hsi_attention + msi_feat

        fused = torch.cat([hsi_enhanced, msi_enhanced], dim=1)
        fused = self.fusion_conv(fused)
        return fused


class LCZ_HMSSNet(nn.Module):
    """
    LCZ classifier with optional HSI, MSI/Sentinel-2, and LST modalities.

    Input channel order:
      both + LST: [HSI bands, MSI bands, LST bands]
      hsi  + LST: [HSI bands, LST bands]
      msi  + LST: [MSI bands, LST bands]

    Tensor shape is always [B, 1, spectral_size, H, W]. LST is off (use_lst=False,
    lst_bands=0) by default; pass lst_bands>0 and use_lst=True to enable it.
    """

    def __init__(
        self,
        in_channels=1,
        spectral_size=25,
        num_classes=17,
        num_tokens=4,
        dim=64,
        modal_type="both",
        hsi_bands=15,
        msi_bands=10,
        lst_bands=0,
        use_lst=False,
        head_dropout=0.3,
    ):
        super().__init__()
        self.L = num_tokens
        self.cT = dim

        self.modal_type = modal_type.lower()
        self.hsi_bands = int(hsi_bands)
        self.msi_bands = int(msi_bands)
        self.lst_bands = int(lst_bands)
        self.use_lst = bool(use_lst or self.lst_bands > 0)
        if self.use_lst and self.lst_bands <= 0:
            raise ValueError("use_lst=True requires lst_bands > 0")

        if self.modal_type == "both":
            expected = self.hsi_bands + self.msi_bands
        elif self.modal_type == "hsi":
            expected = self.hsi_bands
        elif self.modal_type == "msi":
            expected = self.msi_bands
        else:
            raise ValueError("modal_type must be 'both', 'hsi', or 'msi'")

        if self.use_lst:
            expected += self.lst_bands
        assert spectral_size == expected, f"spectral_size should be {expected}, got {spectral_size}"

        self.hsi_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        hsi_2d_in = 8 * self.hsi_bands
        self.hsi_conv2d = nn.Sequential(
            nn.Conv2d(hsi_2d_in, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.hsi_attention = LESA(dim=64, heads=4, dim_head=16)

        self.msi_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
        msi_2d_in = 8 * (self.msi_bands - 1)
        self.msi_conv2d = nn.Sequential(
            nn.Conv2d(msi_2d_in, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.msi_attention = LESA(dim=64, heads=4, dim_head=16)

        self.cross_modal_interaction = CrossModalInteraction(64)

        if self.use_lst:
            self.lst_branch = nn.Sequential(
                nn.Conv2d(self.lst_bands, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.lst_fusion = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        else:
            self.lst_branch = None
            self.lst_fusion = None

        self.token_wAx = nn.Parameter(torch.empty(self.L, 64))
        nn.init.xavier_normal_(self.token_wAx)
        self.token_wVx = nn.Parameter(torch.empty(64, self.cT))
        nn.init.xavier_normal_(self.token_wVx)

        self.token_wAy = nn.Parameter(torch.empty(self.L, 64))
        nn.init.xavier_normal_(self.token_wAy)
        self.token_wVy = nn.Parameter(torch.empty(64, self.cT))
        nn.init.xavier_normal_(self.token_wVy)

        self.head_dropout = nn.Dropout(head_dropout)
        self.nn1 = nn.Linear(dim * self.L, num_classes)
        nn.init.xavier_uniform_(self.nn1.weight)
        nn.init.normal_(self.nn1.bias, std=1e-6)

        self.nn2 = nn.Linear(dim * self.L, num_classes)
        nn.init.xavier_uniform_(self.nn2.weight)
        nn.init.normal_(self.nn2.bias, std=1e-6)

        self.conv2d_features1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[2, 1], stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def _with_attention(self, feat, attention):
        tokens = rearrange(feat, "b c h w -> b (h w) c")
        out = attention(tokens, hw=(feat.shape[-2], feat.shape[-1])) + tokens
        return rearrange(out, "b (h w) c -> b c h w", h=feat.shape[-2], w=feat.shape[-1])

    def forward(self, x):
        if self.use_lst:
            lst_data = x[:, :, -self.lst_bands:, :, :]
            x_main = x[:, :, :-self.lst_bands, :, :]
        else:
            lst_data = None
            x_main = x

        if self.modal_type == "both":
            hsi_data = x_main[:, :, : self.hsi_bands, :, :]
            msi_data = x_main[:, :, self.hsi_bands : self.hsi_bands + self.msi_bands, :, :]
        elif self.modal_type == "hsi":
            hsi_data = x_main
            msi_data = None
        elif self.modal_type == "msi":
            hsi_data = None
            msi_data = x_main
        else:
            raise ValueError("modal_type must be 'both', 'hsi', or 'msi'")

        if hsi_data is not None:
            hsi_3d = self.hsi_conv3d(hsi_data)
            hsi_2d_input = rearrange(hsi_3d, "b c s h w -> b (c s) h w")
            hsi_feat = self.hsi_conv2d(hsi_2d_input)
            hsi_feat = self._with_attention(hsi_feat, self.hsi_attention)
        else:
            hsi_feat = None

        if msi_data is not None:
            msi_3d = self.msi_conv3d(msi_data)
            msi_2d_input = rearrange(msi_3d, "b c s h w -> b (c s) h w")
            msi_feat = self.msi_conv2d(msi_2d_input)
            msi_feat = self._with_attention(msi_feat, self.msi_attention)
        else:
            msi_feat = None

        if self.modal_type == "both":
            x = self.cross_modal_interaction(hsi_feat, msi_feat)
        elif self.modal_type == "hsi":
            x = hsi_feat
        elif self.modal_type == "msi":
            x = msi_feat

        if self.use_lst:
            lst_2d_input = rearrange(lst_data, "b c s h w -> b (c s) h w")
            lst_feat = self.lst_branch(lst_2d_input)
            x = self.lst_fusion(torch.cat([x, lst_feat], dim=1))

        x = F.adaptive_avg_pool2d(x, (9, 9))

        y = rearrange(x, "b c h w -> b c (h w)")
        x_tok = rearrange(x, "b c h w -> b (h w) c")

        wax = rearrange(self.token_wAx, "h w -> w h")
        Ax = torch.einsum("bij,jk->bik", x_tok, wax)
        Ax = rearrange(Ax, "b h w -> b w h").softmax(dim=-1)
        VVx = torch.einsum("bij,jk->bik", x_tok, self.token_wVx)
        Tx = torch.einsum("bij,bjk->bik", Ax, VVx)
        x_tok = rearrange(Tx, "b h w -> b (h w)")
        x_tok = self.nn1(self.head_dropout(x_tok)).unsqueeze(1)

        way = self.token_wAy
        Ay = torch.einsum("ij,bjk->bik", way, y).softmax(dim=-1)
        VVy = torch.einsum("ij,bjk->bik", self.token_wVy, y)
        VVy = rearrange(VVy, "b h w -> b w h")
        Ty = torch.einsum("bij,bjk->bik", Ay, VVy)
        y_tok = rearrange(Ty, "b h w -> b (h w)")
        y_tok = self.nn2(self.head_dropout(y_tok)).unsqueeze(1)

        x_out = torch.cat((x_tok, y_tok), dim=1)
        x_out = x_out.unsqueeze(1)
        x_out = self.conv2d_features1(x_out)
        x_out = x_out.squeeze(1).squeeze(1)
        return x_out


if __name__ == "__main__":
    model_both = LCZ_HMSSNet(spectral_size=25, modal_type="both", hsi_bands=15, msi_bands=10)
    x_both = torch.randn(4, 1, 25, 32, 32)
    print("both output:", model_both(x_both).shape)

    model_hsi = LCZ_HMSSNet(spectral_size=15, modal_type="hsi", hsi_bands=15, msi_bands=10)
    x_hsi = torch.randn(4, 1, 15, 32, 32)
    print("hsi output:", model_hsi(x_hsi).shape)

    model_msi = LCZ_HMSSNet(spectral_size=10, modal_type="msi", hsi_bands=15, msi_bands=10)
    x_msi = torch.randn(4, 1, 10, 32, 32)
    print("msi output:", model_msi(x_msi).shape)

    model_lst = LCZ_HMSSNet(
        spectral_size=26,
        modal_type="both",
        hsi_bands=15,
        msi_bands=10,
        lst_bands=1,
        use_lst=True,
    )
    x_lst = torch.randn(4, 1, 26, 32, 32)
    print("both+lst output:", model_lst(x_lst).shape)
