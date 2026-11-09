#!/usr/bin/env python3
"""生成草稿头子词表（QDV1）：前缀高频区 + 指定文种的全部 token + 特殊 token。

用法:
  python tools/make_draft_vocab.py --tokenizer models/tokenizer/tokenizer.json \
      --out models/draft-vocab-qdv1.bin [--prefix 32768] [--scripts cjk,kana,hangul]

QDV1 格式: u32 magic 0x31564451('QDV1') + u32 n + n 个升序 u32 全局 id。
动机见 docs/SPEC-INVESTIGATION-20261109.md §2/§6：前缀截断（QWENOX_DRAFT_VOCAB）
把高 id 区的中文 token 排除在草稿候选外；子词表按文种覆盖，不偏置语言。
"""
import argparse, json, struct

CJK = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF)]
KANA = [(0x3040, 0x30FF)]
HANGUL = [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]
CYRILLIC = [(0x0400, 0x04FF)]
ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F)]
SCRIPTS = {"cjk": CJK, "kana": KANA, "hangul": HANGUL,
           "cyrillic": CYRILLIC, "arabic": ARABIC}


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(cs, bs))


B2U = bytes_to_unicode()


def decode_tok(s):
    return bytes(B2U[ord(c)] for c in s).decode("utf-8", errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", type=int, default=32768)
    ap.add_argument("--scripts", default="cjk,kana,hangul")
    ap.add_argument("--vocab-size", type=int, default=248320)
    args = ap.parse_args()

    ranges = []
    for name in args.scripts.split(","):
        name = name.strip()
        if name:
            ranges += SCRIPTS[name]

    tk = json.load(open(args.tokenizer, encoding="utf-8"))
    ids = set(range(args.prefix))
    for tok, idx in tk["model"]["vocab"].items():
        text = decode_tok(tok)
        if any(lo <= ord(ch) <= hi for ch in text for lo, hi in ranges):
            ids.add(idx)
    for entry in tk.get("added_tokens", []):
        ids.add(entry["id"])
    ids = sorted(i for i in ids if 0 <= i < args.vocab_size)

    with open(args.out, "wb") as f:
        f.write(struct.pack("<II", 0x31564451, len(ids)))
        f.write(struct.pack(f"<{len(ids)}I", *ids))
    print(f"{args.out}: {len(ids)} ids ({len(ids) / args.vocab_size * 100:.1f}% of vocab)")


if __name__ == "__main__":
    main()
