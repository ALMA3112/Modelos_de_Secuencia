"""Encoder-decoder E1 (LSTM), E2 (GRU) y E3 (GRU + atencion aditiva)."""
import torch
import torch.nn as nn

N_VOICES = 4
VOCAB = 128
CONTEXT = 32
HORIZON = 16


class AdditiveAttention(nn.Module):
    """Atencion aditiva (Bahdanau): score = v^T tanh(Wq q + Wk k)."""

    def __init__(self, hidden):
        super().__init__()
        self.Wq = nn.Linear(hidden, hidden, bias=False)
        self.Wk = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, query, keys):
        # query (B,H), keys (B,S,H) -> contexto (B,H), pesos (B,S)
        scores = self.v(torch.tanh(self.Wq(query).unsqueeze(1) + self.Wk(keys))).squeeze(-1)
        w = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(w.unsqueeze(1), keys).squeeze(1)
        return ctx, w


class BachSeq2Seq(nn.Module):
    def __init__(self, cell="gru", attention=False, hidden=128, emb=16,
                 layers=1, vocab=VOCAB, n_voices=N_VOICES):
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError("cell debe ser 'lstm' o 'gru'")
        self.cell, self.attention = cell, attention
        self.hidden, self.emb_dim, self.layers = hidden, emb, layers
        self.vocab, self.n_voices = vocab, n_voices
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        in_dim = n_voices * emb
        self.embed = nn.Embedding(vocab, emb)
        self.encoder = rnn(in_dim, hidden, num_layers=layers, batch_first=True)
        self.decoder = rnn(in_dim + (hidden if attention else 0), hidden,
                           num_layers=layers, batch_first=True)
        if attention:
            self.attn = AdditiveAttention(hidden)
        self.proj = nn.Linear(hidden * (2 if attention else 1), n_voices * vocab)

    # ---------- configuracion ----------
    @property
    def arch(self):
        return "gru_attn" if (self.cell == "gru" and self.attention) else self.cell

    def get_config(self):
        return dict(arch=self.arch, cell=self.cell, attention=self.attention,
                    hidden=self.hidden, emb=self.emb_dim, layers=self.layers,
                    vocab=self.vocab, n_voices=self.n_voices)

    @classmethod
    def from_config(cls, cfg):
        return cls(cell=cfg["cell"], attention=cfg["attention"], hidden=cfg["hidden"],
                   emb=cfg["emb"], layers=cfg["layers"], vocab=cfg.get("vocab", VOCAB),
                   n_voices=cfg.get("n_voices", N_VOICES))

    # ---------- bloques ----------
    def embed_seq(self, x):
        """(B,T,4) enteros -> (B,T,4*emb), voces concatenadas en orden."""
        return self.embed(x).flatten(2)

    def encode(self, x):
        out, state = self.encoder(self.embed_seq(x))
        return out, state

    @staticmethod
    def _top(state):
        h = state[0] if isinstance(state, tuple) else state
        return h[-1]

    def _step(self, prev, state, enc_out):
        """Un paso del decoder. prev (B,4) -> logits (B,4,128), nuevo estado, pesos."""
        e = self.embed_seq(prev.unsqueeze(1)).squeeze(1)
        ctx = w = None
        if self.attention:
            ctx, w = self.attn(self._top(state), enc_out)
            e = torch.cat([e, ctx], dim=-1)
        out, state = self.decoder(e.unsqueeze(1), state)
        out = out.squeeze(1)
        if self.attention:
            out = torch.cat([out, ctx], dim=-1)
        return self.proj(out).view(-1, self.n_voices, self.vocab), state, w

    # ---------- entrenamiento (teacher forcing) ----------
    def forward(self, x, y, return_attention=False):
        enc_out, state = self.encode(x)
        dec_in = torch.cat([x[:, -1:], y[:, :-1]], dim=1)  # ultimo paso de X + Y[:15]
        if self.attention:
            logits, ws = [], []
            for k in range(dec_in.size(1)):
                lg, state, w = self._step(dec_in[:, k], state, enc_out)
                logits.append(lg)
                ws.append(w)
            logits, attn = torch.stack(logits, 1), torch.stack(ws, 1)
        else:
            out, _ = self.decoder(self.embed_seq(dec_in), state)
            logits = self.proj(out).view(out.size(0), out.size(1), self.n_voices, self.vocab)
            attn = None
        return (logits, attn) if return_attention else logits

    # ---------- generacion libre ----------
    def generate(self, context, steps, temperature=1.0, generator=None,
                 return_attention=False):
        """Devuelve solo los pasos nuevos (B,steps,4). Sin teacher forcing."""
        if temperature <= 0:
            raise ValueError("temperature debe ser > 0")
        if steps < 1:
            raise ValueError("steps debe ser >= 1")
        squeeze = context.dim() == 2
        if squeeze:
            context = context.unsqueeze(0)
        context = context.long()
        if context.size(1) < CONTEXT or context.size(2) != self.n_voices:
            raise ValueError(f"el contexto debe tener forma (B,>={CONTEXT},{self.n_voices})")

        was_training = self.training
        self.eval()
        new, attn_blocks, ctx = [], [], context
        try:
            with torch.no_grad():
                remaining = steps
                while remaining > 0:
                    n = min(HORIZON, remaining)
                    enc_out, state = self.encode(ctx[:, -CONTEXT:])  # reinicia el decoder
                    prev = ctx[:, -1]
                    block, ws = [], []
                    for _ in range(n):
                        logits, state, w = self._step(prev, state, enc_out)
                        probs = torch.softmax(logits / temperature, dim=-1)
                        prev = torch.multinomial(probs.reshape(-1, self.vocab), 1,
                                                 generator=generator).view(-1, self.n_voices)
                        block.append(prev)
                        if w is not None:
                            ws.append(w)
                    block = torch.stack(block, 1)
                    new.append(block)
                    if ws:
                        attn_blocks.append(torch.stack(ws, 1))
                    ctx = torch.cat([ctx, block], dim=1)
                    remaining -= n
        finally:
            self.train(was_training)

        gen = torch.cat(new, 1)
        if squeeze:
            gen = gen.squeeze(0)
            attn_blocks = [a.squeeze(0) for a in attn_blocks]
        return (gen, attn_blocks or None) if return_attention else gen


def build_model(arch, hidden=128, emb=16, layers=1):
    cell, attn = {"lstm": ("lstm", False), "gru": ("gru", False),
                  "gru_attn": ("gru", True)}[arch]
    return BachSeq2Seq(cell=cell, attention=attn, hidden=hidden, emb=emb, layers=layers)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
