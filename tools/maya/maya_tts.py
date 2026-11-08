#!/usr/bin/env python
"""Standalone Slovenian TTS CLI for the Maya1 phoneme model (like the Piper CLI, but for Maya).

Loads base maya-research/maya1 + the Slovenian LoRA adapter, phonemizes input with espeak-ng "sl",
generates SNAC-24kHz codec tokens, decodes to a wav. Self-contained: only needs the pip deps and an
espeak-ng shared library. Emotion presets pick a voice description the model understands.

  # neutral
  .venv/bin/python scripts/maya_tts.py -t "Danes je lep sončen dan." -o out.wav
  # with emotion
  .venv/bin/python scripts/maya_tts.py -t "To je čudovito!" -e veselo -o out.wav
  # from a file, custom adapter (HF id or local dir)
  .venv/bin/python scripts/maya_tts.py -f besedilo.txt -a texdata/Maya1-Slovenian-TTS -o out.wav

Deps: pip install transformers peft torch snac soundfile phonemizer
"""
from __future__ import annotations

import argparse
import os
import re
import sys

os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")

CODE_START, CODE_END, OFFSET = 128257, 128258, 128266
SNAC_MIN, SNAC_MAX = 128266, 156937
DEFAULT_ADAPTER = "outputs/maya_ms_phon_sl"
DEFAULT_SPEAKER = "Artur-B-G0911"
_TAG_RE = re.compile(r"<[a-z_]+>")

# emotion word must match what the model was trained on (EmoV-DB mix); described in English but
# applied to Slovenian phonemes (cross-lingual transfer)
EMOTIONS = {
    "nevtralno": "A neutral Slovenian voice.",
    "veselo":    "An amused, cheerful, laughing Slovenian voice.",
    "jezno":     "An angry, upset Slovenian voice.",
    "gnus":      "A disgusted Slovenian voice.",
    "zaspano":   "A sleepy, tired Slovenian voice.",
}


def phonemize_sl(text, backend):
    from phonemizer.separator import Separator
    sep = Separator(phone="", word=" ")
    parts, tags, out = _TAG_RE.split(text), _TAG_RE.findall(text), []
    for i, chunk in enumerate(parts):
        chunk = chunk.strip()
        if chunk:
            out.append(backend.phonemize([chunk], separator=sep, strip=True)[0])
        if i < len(tags):
            out.append(tags[i])
    return " ".join(out).strip()


def unpack_snac(ids):
    frames = len(ids) // 7
    ids = ids[:frames * 7]
    l1, l2, l3 = [], [], []
    for i in range(frames):
        s = ids[i * 7:(i + 1) * 7]
        l1.append((s[0] - OFFSET) % 4096)
        l2 += [(s[1] - OFFSET) % 4096, (s[4] - OFFSET) % 4096]
        l3 += [(s[2] - OFFSET) % 4096, (s[3] - OFFSET) % 4096, (s[5] - OFFSET) % 4096, (s[6] - OFFSET) % 4096]
    return [l1, l2, l3]


def main():
    ap = argparse.ArgumentParser(description="Maya1 Slovenian TTS (phoneme front-end)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-t", "--text", help="text to speak")
    g.add_argument("-f", "--file", help="read text from a file")
    g.add_argument("--list-emotions", action="store_true")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("-a", "--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter (HF id or local dir)")
    ap.add_argument("-e", "--emotion", choices=list(EMOTIONS), help="emotion preset")
    ap.add_argument("-d", "--desc", help="raw voice description (overrides --emotion)")
    ap.add_argument("-s", "--speaker", default=DEFAULT_SPEAKER)
    ap.add_argument("--base", default="maya-research/maya1")
    ap.add_argument("--temperature", type=float, default=0.75)  # sweet spot: natural without artefacts
    ap.add_argument("--max-new", type=int, default=1800)
    args = ap.parse_args()

    if args.list_emotions:
        print("emotions:")
        for k, v in EMOTIONS.items():
            print(f"  {k:12s} {v}")
        return

    text = args.file and open(args.file, encoding="utf-8").read().strip() or args.text
    desc = args.desc or (EMOTIONS[args.emotion] if args.emotion else f"A Slovenian voice, speaker {args.speaker}.")

    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    from phonemizer.backend import EspeakBackend

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading maya1 + adapter '{args.adapter}' ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(dev)
    if args.adapter and args.adapter.lower() not in ("none", "base"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(dev)
    be = EspeakBackend("sl", with_stress=True, preserve_punctuation=True)

    phonemes = phonemize_sl(text, be)  # REQUIRED: model expects IPA, not graphemes
    content = f'<description="{desc}"> {phonemes}'
    prompt = tok.apply_chat_template([{"role": "user", "content": content}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                             temperature=args.temperature, top_p=0.9, repetition_penalty=1.1,
                             eos_token_id=CODE_END, pad_token_id=CODE_END)
    gen = out[0, ids.shape[1]:].tolist()
    snac_ids = [t for t in gen if SNAC_MIN <= t <= SNAC_MAX]
    if len(snac_ids) < 7:
        print("no audio generated", file=sys.stderr)
        sys.exit(1)
    codes = [torch.tensor(c, device=dev).unsqueeze(0) for c in unpack_snac(snac_ids)]
    with torch.inference_mode():
        audio = snac.decoder(snac.quantizer.from_codes(codes))[0, 0].cpu().numpy()
    sf.write(args.out, audio, 24000)
    print(f"wrote {args.out} ({len(audio)/24000:.2f}s)  desc={desc!r}")


if __name__ == "__main__":
    main()
