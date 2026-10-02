"""
Real Text Dataset & Tokenizer Loader — Progetto 21.0 (FAM-LLM)
==============================================================
Carica i token del dataset GPT-2 con accesso LAZY tramite numpy.memmap:
- Riusa direttamente il corpus pre-tokenizzato di 61.946.938 token da Progetto 20.0
  (openwebtext_large.bin) senza duplicare file e con consumo di RAM pari a zero.
- Supporta il ritaglio istantaneo a qualsiasi numero di token (es. 100k, 1M, 10M o tutti i 62M).
- Accesso O(1) alle finestre autoregressive (x, y) per Next-Token Prediction.
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

DTYPE = np.uint16
DTYPE_BYTES = 2


class GPT2TokenDataset(Dataset):
    """
    Dataset memmap ad alte prestazioni per Progetto 21.0:
    - Zero RAM overhead: legge i token direttamente dal disco on-demand.
    - Riuso immediato di openwebtext_large.bin (61.946.938 token).
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        seq_len: int = 256,
        max_tokens: int | None = None,
    ):
        self.seq_len = seq_len
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab  # 50257

        resolved_path = self._resolve_dataset_path(data_path)
        self.bin_path = resolved_path

        # Calcola numero totale di token nel file binario
        total_tokens_in_file = self.bin_path.stat().st_size // DTYPE_BYTES

        if max_tokens and max_tokens > 0:
            self.n_tokens = min(total_tokens_in_file, max_tokens)
        else:
            self.n_tokens = total_tokens_in_file

        # Apertura memmap (read-only, accesso O(1) on-demand)
        self.data = np.memmap(self.bin_path, dtype=DTYPE, mode="r", shape=(total_tokens_in_file,))
        self.tokens_view = self.data[:self.n_tokens]

        if self.n_tokens < self.seq_len + 1:
            raise ValueError(f"Il corpus ha solo {self.n_tokens} token, ma seq_len={seq_len} richiede almeno {seq_len + 1}.")

        self._num_samples = (self.n_tokens - 1) // self.seq_len
        print(f"📊 Dataset MemMap pronto: {self.n_tokens:,} token attivi (su {total_tokens_in_file:,} disponibili)")
        print(f"📦 Finestre di addestramento: {self._num_samples:,} campioni | seq_len={seq_len}")

    def _resolve_dataset_path(self, data_path: str | Path | None) -> Path:
        if data_path:
            p = Path(data_path)
            if p.exists() and p.stat().st_size > 0:
                return p

        # 1. Cerca il file openwebtext_large.bin nella cartella data di progetto21.0 o progetto20.0
        workspace_root = Path(__file__).parent.parent.parent
        candidate_paths = [
            Path(__file__).parent.parent / "data" / "openwebtext_large.bin",
            workspace_root / "progetto20.0" / "data" / "openwebtext_large.bin",
            Path(__file__).parent.parent / "data" / "gpt2_100k_tokens.bin",
        ]

        for cand in candidate_paths:
            if cand.exists() and cand.stat().st_size > 1000:
                print(f"⚡ Riuso corpus binario GPT-2: {cand}")
                return cand

        raise FileNotFoundError("Nessun dataset binario trovato tra i percorsi noti.")

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        # Legge solo la finestra necessaria (seq_len + 1)
        chunk = self.tokens_view[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


# Alias per retrocompatibilità
RealTextDataset = GPT2TokenDataset


def get_dataloader(
    data_path: str | None = None,
    seq_len: int = 256,
    batch_size: int = 4,
    shuffle: bool = True,
    max_tokens: int | None = None,
    num_workers: int = 0,
):
    dataset = GPT2TokenDataset(data_path=data_path, seq_len=seq_len, max_tokens=max_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader, dataset
