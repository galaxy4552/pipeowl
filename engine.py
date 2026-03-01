from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np  # type: ignore
from pathlib import Path

BASE_DIR = Path(__file__).resolve()
DATA_DIR = BASE_DIR.parent / "data"

## 我只有推理和抽象成更好的架構 程式碼還是由AI完成

@dataclass
class PipeOwlConfig:
    """
    全域設定。

    embeddings_path:
        語義場的基底向量矩陣 (V, D)
        V = 詞彙數
        D = 向量維度

    delta_scalar_path:
        每個 token 對應的一維場偏移量 (V,)
        用來做 score 偏移（目前為靜態 bias）

    vocab_path:
        vocab list，必須與 embeddings 順序完全對齊。
        index i <-> emb[i] <-> delta[i]

    alpha:
        base 相似度權重

    beta:
        delta 權重（目前為 logit bias，不是動態 loss）

    top_k:
        retrieval 預設回傳數量

    temperature:
        decode 階段採樣溫度

    max_new_tokens:
        decode 最大生成長度
    """
    embeddings_path: str = str(DATA_DIR / "L1_base_embeddings.npy")
    delta_scalar_path: str = str(DATA_DIR / "delta_base_scalar.npy")
    vocab_path: str = str(DATA_DIR / "L1_base_vocab.json")

    # scoring: score = alpha * base_sim + beta * delta_scalar
    alpha: float = 1.0
    beta: float = 0.0


    # retrieval
    top_k: int = 16

    # decode
    temperature: float = 0.8
    max_new_tokens: int = 64

## semanticizer 
class VocabTokenizer:
    """
    字串最大匹配 tokenizer。

    設計目標：
        將輸入文字拆成 vocab 中存在的 token。

    方法：
        - 使用最大長度優先匹配
        - OOV 字元直接跳過

    風險：
        - OOV 會被忽略（可能導致語義缺失）
        - 無 subword fallback
        - 時間複雜度 O(n * max_token_len)

    適用情境：
        vocab 是字 / 詞 級別，且已對齊 embedding。
    """
    def __init__(self, vocab_list):
        self.vocab_set = set(vocab_list)
        self.max_len = max(len(t) for t in vocab_list)

    def tokenize(self, text: str):
        tokens = []
        i = 0
        n = len(text)

        while i < n:
            matched = False
            for L in range(self.max_len, 0, -1):
                if i + L <= n:
                    piece = text[i:i+L]
                    if piece in self.vocab_set:
                        tokens.append(piece)
                        i += L
                        matched = True
                        break
            if not matched:
                i += 1  # 跳過 OOV
        return tokens

class PipeOwlEngine:
    """
    PipeOwl 幾何語義引擎核心。

    設計哲學：
        index = 語義場座標

        emb[i]     -> 詞向量
        delta[i]   -> 詞的場偏移量
        vocab[i]   -> 詞本身

    核心流程：
        text
          ↓
        tokenize
          ↓
        mean embedding
          ↓
        score = alpha*base + beta*delta
          ↓
        top-k
          ↓
        decode

    這是一個：
        Field-based retrieval language system
    """

    def __init__(self, cfg: PipeOwlConfig):
        self.cfg = cfg

        # Loaded assets
        self.emb: np.ndarray = None            # (V, D) float32
        self.delta: np.ndarray = None          # (V,) float32
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []

        # Decoder (optional)
        self.decoder = MicroGPTDecoder()       # inference-only stub; plug your trained weights later

        self._load_assets()

    # -------------------------
    # asset loading
    # -------------------------

    def _load_assets(self) -> None:
        """
        載入語義場資產。

        載入內容：
            1. embeddings (V, D)
            2. delta scalar (V,)
            3. vocab list (V,)

        關鍵假設：
            三者必須 index 完全對齊。

        幾何意義：
            每個 index i 對應語義空間中的一個固定場點。

        風險：
            - vocab 長度不等於 embeddings
            - delta 長度不等於 embeddings
            - dtype 不一致
        """
        if not os.path.exists(self.cfg.embeddings_path):
            raise FileNotFoundError(self.cfg.embeddings_path)
        if not os.path.exists(self.cfg.delta_scalar_path):
            raise FileNotFoundError(self.cfg.delta_scalar_path)
        if not os.path.exists(self.cfg.vocab_path):
            raise FileNotFoundError(self.cfg.vocab_path)

        # embeddings: (V, D)
        self.emb = np.load(self.cfg.embeddings_path)
        if self.emb.dtype != np.float32:
            self.emb = self.emb.astype(np.float32, copy=False)

        # delta: (V,)
        self.delta = np.load(self.cfg.delta_scalar_path)
        if self.delta.dtype != np.float32:
            self.delta = self.delta.astype(np.float32, copy=False)

        if self.emb.ndim != 2:
            raise ValueError(f"embeddings must be 2D (V, D), got shape={self.emb.shape}")
        V, D = self.emb.shape

        if self.delta.ndim != 1 or self.delta.shape[0] != V:
            raise ValueError(f"delta must be shape (V,), got {self.delta.shape}, expected ({V},)")

        # vocab json: build token_to_id and id_to_token
        with open(self.cfg.vocab_path, "r", encoding="utf-8") as f:
            vocab_list = json.load(f)

        if not isinstance(vocab_list, list):
            raise ValueError("vocab must be a list for geometric field mode")

        if len(vocab_list) != V:
            raise ValueError(f"vocab size {len(vocab_list)} != embeddings V {V}")

        self.vocab = vocab_list
        self.id_to_token = vocab_list
        self.token_to_id = {ch: i for i, ch in enumerate(vocab_list)}

        self.tokenizer = VocabTokenizer(self.vocab)

    # -------------------------
    # encode (from vector library)
    # -------------------------

    def encode(self, text: str):
        """
        將文字投影到語義場中。

        流程：
            1. tokenize -> token list
            2. 取每個 token 對應 emb
            3. 做 mean pooling
            4. normalize

        數學形式：
            q = normalize( mean( emb[token_i] ) )

        幾何意義：
            這是在語義場中求質心。

        風險：
            - mean pooling 會削弱方向性
            - 若 tokens 少或 OOV 多，向量會接近零
        """
        tokens = self.tokenizer.tokenize(text)

        vecs = []
        for t in tokens:
           idx = self.token_to_id[t]
           vecs.append(self.emb[idx])

        if not vecs:
            return np.zeros(self.emb.shape[1], dtype=np.float32)

        q = np.mean(vecs, axis=0)
        q /= (np.linalg.norm(q) + 1e-12)
        return q

    # -------------------------
    # loss / scoring (delta)
    # -------------------------
    def score_vocab(self, q: np.ndarray, alpha: Optional[float] = None, beta: Optional[float] = None) -> np.ndarray:
        """
        計算每個 vocab token 的場分數。

        base:
            emb @ q
            若 emb 與 q 已正規化，則為 cosine similarity。

        delta:
            每個 token 的靜態場偏移量。

        最終公式：
            score = alpha * base + beta * delta

        目前語義：
            delta 是 logit bias。
            不是 loss、不是 energy gradient。

        暫無實作
        若 beta = 0：
            純 embedding 相似度搜尋。

        若 beta > 0：
            加入場重力井效果。
        """
        a = self.cfg.alpha if alpha is None else float(alpha)
        b = self.cfg.beta if beta is None else float(beta)

        base = self.emb @ q  # (V,)
        score = a * base + b * self.delta
        return score.astype(np.float32, copy=False)

    def topk(self, score: np.ndarray, k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        取前 k 高分 token。

        使用 argpartition 提升效率。

        回傳：
            [(token_string, score), ...]

        幾何意義：
            找出最接近 query 向量（含場偏移）的場點。

        注意：
            score 可能 > 1（因為加入 delta）。
        """
        k = self.cfg.top_k if k is None else int(k)
        k = max(1, min(k, score.shape[0]))

        # argpartition for speed
        idx = np.argpartition(-score, k - 1)[:k]
        idx = idx[np.argsort(-score[idx])]

        out = []
        for i in idx:
            tok = self.id_to_token[i] if i < len(self.id_to_token) else str(i)
            out.append((tok, float(score[i])))
        return out

    # -------------------------
    # decode (microgpt inference-only)
    # -------------------------
    def decode(self, prompt_tokens: List[str]) -> str:
        """
        Decode 階段。

        目前行為：
            將 top tokens 拼成 prompt 字串，
            丟給 microgpt stub。

        設計定位：
            retrieval 與 generation 分離。

        現狀：
            microgpt 尚未接上真實權重，
            目前只是 pipeline 占位。
        """

        prompt = " ".join([t for t in prompt_tokens if t])
        return self.decoder.generate(
            prompt=prompt,
            temperature=self.cfg.temperature,
            max_new_tokens=self.cfg.max_new_tokens,
        )

    # -------------------------
    # one-shot pipeline
    # -------------------------
    def pipeowl(
        self,
        text: str,
        *,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        單次完整 pipeline。

        流程：
            text
              ↓
            encode
              ↓
            score_vocab
              ↓
            topk
              ↓
            decode

        回傳：
            {
                "query": 原始文字,
                "retrieved": top-k token + 分數,
                "prompt": 用於 decode 的 token 串,
                "decoded": 生成結果
            }

        這是語義場查詢的一次完整觀測。
        """
        q = self.encode(text)
        s = self.score_vocab(q, alpha=alpha, beta=beta)
        retrieved = self.topk(s, k=top_k)

        # build a prompt from top tokens (simple & deterministic)
        prompt_tokens = [t for (t, _) in retrieved[: min(len(retrieved), 8)]]
        if temperature is not None:
            self.cfg.temperature = float(temperature)
        if max_new_tokens is not None:
            self.cfg.max_new_tokens = int(max_new_tokens)

        decoded = self.decode(prompt_tokens)
        return {
            "query": text,
            "retrieved": retrieved,
            "prompt": " ".join(prompt_tokens),
            "decoded": decoded,
        }


# ----------------------------------------------------------------------
# microgpt inference-only stub
# ----------------------------------------------------------------------
class MicroGPTDecoder:
    """
    推理階段占位 decoder。

    設計目的：
        讓 pipeline 可運行，
        未來可替換為：
            - 已訓練 microGPT
            - 外部 LLM
            - 或場驅動 sampling 模型

    現在只是 scaffold。
    
    Inference-only placeholder.

    Why placeholder?
    - Your pasted microGPT file trains its own weights in-process.
    - For a real decode stage, you want:
      (A) load a trained state_dict from disk, OR
      (B) keep a tiny trained model in memory, OR
      (C) use microGPT purely as a sampler over a learned char vocab.

    This class is the stable interface. Plug your implementation later.
    """

    def __init__(self):
        # If you already have trained weights, add:
        # self.state_dict = load(...)
        pass

    def generate(self, prompt: str, temperature: float = 0.8, max_new_tokens: int = 64) -> str:
        # Minimal safe fallback: return prompt as “decoded” scaffold.
        # Replace this with your microgpt forward+sampling once you have weights.
        # (This keeps the pipeline callable today.)
        return f"[microgpt_stub] {prompt}"
