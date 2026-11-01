"""最小 ACT (Zhao et al. 2023): CVAE + Transformer encoder/decoder.

フック用モジュール名(固定):
  encoder_layer{i}_out : encoder 第 i 層の出力 [B, 4, d]
  encoder_out          : encoder 最終出力 [B, 4, d] (トークン順: latent, robot, cube, color)
  decoder_layer{i}_out : decoder 第 i 層の出力 [B, K, d] (最終層 = decoder_out と同じ値)
  decoder_out          : decoder 最終出力 [B, K, d] (action_head の直前)
"""
import torch
import torch.nn as nn

TOKENS = ["latent", "robot", "cube", "color"]


class ACT(nn.Module):
    def __init__(self, robot_dim=9, cube_dim=10, color_dim=4, act_dim=7, chunk=20, d_model=256, heads=8,
                 ffn=1024, enc_layers=4, dec_layers=4, latent=32, dropout=0.1, **_):
        super().__init__()
        self.chunk, self.latent_dim, self.act_dim = chunk, latent, act_dim
        d = d_model
        # CVAE encoder (学習時のみ): [CLS, robot, a_1..a_K]
        self.cvae_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cvae_robot = nn.Linear(robot_dim, d)
        self.cvae_act = nn.Linear(act_dim, d)
        self.cvae_pos = nn.Parameter(torch.randn(1, chunk + 2, d) * 0.02)
        self.cvae_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True), enc_layers)
        self.cvae_out = nn.Linear(d, 2 * latent)
        # 主 encoder: [latent, robot, cube, color]
        self.latent_proj = nn.Linear(latent, d)
        self.robot_proj = nn.Linear(robot_dim, d)
        self.cube_proj = nn.Linear(cube_dim, d)
        self.color_proj = nn.Linear(color_dim, d)
        self.enc_pos = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        self.enc_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True) for _ in range(enc_layers)])
        for i in range(enc_layers):
            self.add_module(f"encoder_layer{i}_out", nn.Identity())
        self.encoder_out = nn.Identity()
        # decoder: K 個の学習可能クエリ
        self.queries = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, ffn, dropout, batch_first=True), dec_layers)
        for i in range(dec_layers):
            self.add_module(f"decoder_layer{i}_out", nn.Identity())
        self.decoder_out = nn.Identity()
        self.action_head = nn.Linear(d, act_dim)

    def encode_latent(self, robot, actions, is_pad):
        B = robot.shape[0]
        x = torch.cat([self.cvae_cls.expand(B, -1, -1), self.cvae_robot(robot)[:, None], self.cvae_act(actions)], 1)
        x = x + self.cvae_pos
        mask = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=robot.device), is_pad], 1)
        h = self.cvae_enc(x, src_key_padding_mask=mask)[:, 0]
        mu, logvar = self.cvae_out(h).chunk(2, -1)
        return mu, logvar

    def encode(self, robot, cube, color, z):
        x = torch.stack([self.latent_proj(z), self.robot_proj(robot), self.cube_proj(cube), self.color_proj(color)], 1)
        x = x + self.enc_pos
        for i, layer in enumerate(self.enc_layers):
            x = layer(x)
            x = getattr(self, f"encoder_layer{i}_out")(x)
        return self.encoder_out(x)

    def decode(self, memory):
        B = memory.shape[0]
        # nn.TransformerDecoder の forward と同じ計算を層ごとに回し, 各層の出力に Identity を挟む
        # (パラメータ名 decoder.layers.{i}.* は変わらないので既存の重みをそのまま読める)
        h = self.queries.expand(B, -1, -1)
        for i, layer in enumerate(self.decoder.layers):
            h = layer(h, memory)
            h = getattr(self, f"decoder_layer{i}_out")(h)
        if self.decoder.norm is not None:
            h = self.decoder.norm(h)
        h = self.decoder_out(h)
        return self.action_head(h)

    def forward(self, robot, cube, color, actions=None, is_pad=None):
        B = robot.shape[0]
        if actions is not None:
            mu, logvar = self.encode_latent(robot, actions, is_pad)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=robot.device)
        return self.decode(self.encode(robot, cube, color, z)), mu, logvar


def act_loss(pred, actions, is_pad, mu, logvar, kl_weight):
    l1 = ((pred - actions).abs().mean(-1) * (~is_pad)).sum() / (~is_pad).sum()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl


def build_act(cfg):
    a = cfg["act"]
    return ACT(chunk=a["chunk"], d_model=a["d_model"], heads=a["heads"], ffn=a["ffn"],
               enc_layers=a["enc_layers"], dec_layers=a["dec_layers"], latent=a["latent"], dropout=a["dropout"])


def load_policy(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_act(ck["cfg"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["cfg"], ck["stats"]
