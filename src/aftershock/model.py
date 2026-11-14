"""Models: a post-image-only baseline and the Siamese change-detection network."""
import torch
import torch.nn as nn
import torchvision

N_CLASSES = 4


def resnet18_encoder(pretrained: bool) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    enc = torchvision.models.resnet18(weights=weights)
    enc.fc = nn.Identity()          # 512-d feature vector
    return enc


def encoder_features(enc: nn.Module, x: torch.Tensor):
    """Run the ResNet stem + stages explicitly; returns (last conv activation, pooled vector) for Grad-CAM."""
    x = enc.maxpool(enc.relu(enc.bn1(enc.conv1(x))))
    a = enc.layer4(enc.layer3(enc.layer2(enc.layer1(x))))
    return a, torch.flatten(enc.avgpool(a), 1)


def head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, N_CLASSES))


class SiameseDamageNet(nn.Module):
    """Shared encoder for pre and post crops; classifies [pre, post, |post - pre|] features."""
    arch = "siamese"

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = resnet18_encoder(pretrained)
        self.head = head(512 * 3, dropout)

    def classify(self, f_pre, f_post):
        return self.head(torch.cat([f_pre, f_post, (f_post - f_pre).abs()], dim=1))

    def forward(self, pre, post):
        return self.classify(encoder_features(self.encoder, pre)[1], encoder_features(self.encoder, post)[1])


class SiameseFMNet(nn.Module):
    """Siamese with feature-map (mid-level) fusion.

    The plain Siamese net pools each image to one vector before comparing, so it loses WHERE things changed.
    Here the shared encoder runs up to layer3 on both crops, the two spatial feature maps are compared
    position by position ([pre, post, |post - pre|] -> 1x1 conv), and layer4 then looks for change patterns
    in that fused map before pooling.
    """
    arch = "siamese_fm"

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = resnet18_encoder(pretrained)                     # stem + layer1-3 are shared
        self.fuse = nn.Sequential(nn.Conv2d(256 * 3, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.head = head(512, dropout)

    def _layer3(self, x):
        e = self.encoder
        return e.layer3(e.layer2(e.layer1(e.maxpool(e.relu(e.bn1(e.conv1(x)))))))

    def fused_forward(self, pre, post):
        """Returns (fused layer4 activation [B,512,h,w] for Grad-CAM, logits)."""
        a_pre, a_post = self._layer3(pre), self._layer3(post)
        act = self.encoder.layer4(self.fuse(torch.cat([a_pre, a_post, (a_post - a_pre).abs()], dim=1)))
        return act, self.head(torch.flatten(self.encoder.avgpool(act), 1))

    def forward(self, pre, post):
        return self.fused_forward(pre, post)[1]


class PostOnlyNet(nn.Module):
    """Baseline: looks only at the post-disaster crop."""
    arch = "post"

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = resnet18_encoder(pretrained)
        self.head = head(512, dropout)

    def classify(self, f_pre, f_post):
        return self.head(f_post)

    def forward(self, pre, post):
        return self.classify(None, encoder_features(self.encoder, post)[1])


def build_model(arch: str, pretrained=True, dropout=0.3) -> nn.Module:
    return {"siamese": SiameseDamageNet, "siamese_fm": SiameseFMNet, "post": PostOnlyNet}[arch](pretrained=pretrained, dropout=dropout)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): down-weights easy examples so rare damage classes get attention."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.ones(N_CLASSES))
        self.gamma = gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        p = logp.exp()
        return (-self.weight[target] * (1 - p) ** self.gamma * logp).mean()


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], pretrained=False, dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
