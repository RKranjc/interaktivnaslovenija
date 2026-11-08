---
license: apache-2.0
base_model: maya-research/maya1
library_name: peft
pipeline_tag: text-to-speech
language:
- sl
tags:
- text-to-speech
- tts
- slovenian
- maya1
- snac
- lora
- phoneme
---

# Maya1 Slovenian TTS (phoneme front-end)

> Trained by **[MediaAtlas](https://mediaatlas.si/ai-training.html)** — LLM fine-tuning on your own data, trained in the EU, weights delivered. [Pricing](https://mediaatlas.si/ai-training-pricing.html) · [All our models](https://huggingface.co/texdata)

A LoRA adapter that adapts [maya-research/maya1](https://huggingface.co/maya-research/maya1) (a 3B
Orpheus-style speech LM that autoregressively generates SNAC-24kHz codec tokens) to **Slovenian**.

The key design choice is a **phoneme front-end**: Maya1 is grapheme-based and English-pretrained, so
it applies English grapheme-to-phoneme rules to Slovenian text (mispronunciations, wrong or missing
stress — Slovenian orthography does not mark stress, word-boundary errors). This adapter is trained on
**espeak-ng `sl` IPA phonemes** instead of raw text, so the model only has to learn phoneme-to-acoustic.
This gives markedly more accurate Slovenian pronunciation.

**You MUST phonemize input text with espeak-ng `sl` before inference** — the model expects IPA, not
graphemes. Feeding raw Slovenian text will not work correctly.

## Benchmark

Synthesize held-out ARTUR validation texts, transcribe back with a Slovenian ASR model, compare (jiwer):

| model | WER | CER |
|---|---|---|
| grapheme, clean multi-speaker | 21.8% | 5.0% |
| **phoneme (this model)** | 24.8% | **4.2%** |

CER (character error rate) is the cleaner phonetic-accuracy metric and is the lowest of every variant
we trained — the phoneme front-end measurably improves pronunciation. (WER is more sensitive to the ASR
model and word boundaries; the difference is within noise at n=15.)

## How it was made

- **Base:** maya-research/maya1 (3B `LlamaForCausalLM`, generates SNAC-24kHz tokens, Apache-2.0).
- **Data:** [RSDO/ARTUR](https://www.clarin.si/repository/xmlui/handle/11356/1776) read Slovenian speech.
  Clean multi-speaker selection: keep only speakers with >= 25 eligible utterances (drops single-clip
  pseudo-speakers), 20000 utterances across ~1300 speakers, 0.8-14s, per-speaker voice descriptions
  (`A Slovenian voice, speaker {id}.`).
- **Audio -> tokens:** resample to 24kHz mono, encode with [SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz)
  (7 tokens/frame, packed to Maya1's `128266 + pos*4096 + code` layout).
- **Text -> phonemes:** espeak-ng `sl` via the `phonemizer` library, `with_stress=True`.
- **Training:** LoRA (rank 128, alpha 256) on `q,k,v,o,gate,up,down`, 2 epochs, LR 1e-4, bf16,
  gradient checkpointing. Target = `[CODE_START] + snac_ids + [CODE_END]`, loss only on the audio side.

## Quick start (CLI)

A self-contained CLI [`maya_tts.py`](maya_tts.py) is included in this repo — it handles phonemization,
generation and decoding for you:

```bash
pip install transformers peft torch snac soundfile phonemizer   # + an espeak-ng shared lib
huggingface-cli download texdata/Maya1-Slovenian-TTS maya_tts.py --local-dir .

python maya_tts.py -t "Danes je lep sončen dan." -a texdata/Maya1-Slovenian-TTS -o out.wav
python maya_tts.py -t "To je čudovito!" -a texdata/Maya1-Slovenian-TTS -e veselo -o out.wav
python maya_tts.py --list-emotions
```

## Usage (library)

```bash
pip install transformers peft torch snac soundfile phonemizer
# espeak-ng shared lib must be present (Debian/Ubuntu: apt-get install espeak-ng-data,
# or set PHONEMIZER_ESPEAK_LIBRARY to your libespeak-ng.so)
```

```python
import os, torch, soundfile as sf
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from snac import SNAC
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

CODE_START, CODE_END, OFFSET = 128257, 128258, 128266
SNAC_MIN, SNAC_MAX = 128266, 156937
dev = "cuda"

tok = AutoTokenizer.from_pretrained("maya-research/maya1")
model = AutoModelForCausalLM.from_pretrained("maya-research/maya1", dtype=torch.bfloat16).to(dev)
model = PeftModel.from_pretrained(model, "texdata/Maya1-Slovenian-TTS").eval()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(dev)
be = EspeakBackend("sl", with_stress=True, preserve_punctuation=True)

def unpack_snac(ids):
    frames = len(ids) // 7; ids = ids[:frames*7]; l1,l2,l3 = [],[],[]
    for i in range(frames):
        s = ids[i*7:(i+1)*7]
        l1.append((s[0]-OFFSET)%4096)
        l2 += [(s[1]-OFFSET)%4096, (s[4]-OFFSET)%4096]
        l3 += [(s[2]-OFFSET)%4096, (s[3]-OFFSET)%4096, (s[5]-OFFSET)%4096, (s[6]-OFFSET)%4096]
    return [l1,l2,l3]

def synth(text, desc="A Slovenian voice, speaker Artur-B-G0911.", temperature=0.75):  # 0.75 = natural sweet spot
    phonemes = be.phonemize([text], separator=Separator(phone="", word=" "), strip=True)[0]  # REQUIRED
    content = f'<description="{desc}"> {phonemes}'
    prompt = tok.apply_chat_template([{"role": "user", "content": content}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    out = model.generate(ids, max_new_tokens=1800, do_sample=True, temperature=temperature,
                         top_p=0.9, repetition_penalty=1.1, eos_token_id=CODE_END, pad_token_id=CODE_END)
    gen = out[0, ids.shape[1]:].tolist()
    snac_ids = [t for t in gen if SNAC_MIN <= t <= SNAC_MAX]
    codes = [torch.tensor(c, device=dev).unsqueeze(0) for c in unpack_snac(snac_ids)]
    audio = snac.decoder(snac.quantizer.from_codes(codes))[0,0].detach().cpu().numpy()
    return audio

sf.write("out.wav", synth("Danes je lep sončen dan in ptice pojejo."), 24000)
```

## Notes

- **Emotions:** the base Maya1 supports expressive tags (`<laugh>`, `<sigh>`, ...) and emotional voice
  descriptions in English. This adapter was fine-tuned on neutral read ARTUR speech, so Slovenian
  emotional control is limited; expect neutral prosody. Cross-lingual emotional fine-tuning is future work.
- `repetition_penalty=1.1` is important — it prevents non-termination (runaway generation with no
  `CODE_END`) on short inputs.
- `max_new_tokens` must cover your longest utterance (~1800 for ~10s) or the audio truncates.

## License

Apache-2.0, inherited from the base model maya-research/maya1.

## Citation

**Cite this work** (Tadej Fius, MediaAtlas Ltd):

```bibtex
@misc{fius2026maya,
  title        = {Maya1 Slovenian TTS},
  author       = {Fius, Tadej},
  year         = {2026},
  publisher    = {MediaAtlas Ltd},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/texdata/Maya1-Slovenian-TTS}
}
```

Upstream / source citations:

```bibtex
@misc{maya1, title={Maya1 (Orpheus-style speech LM)}, author={{Maya Research}}, note={Apache-2.0}, url={https://huggingface.co/maya-research/maya1}}
@misc{siuzdak2024snac, title={SNAC: Multi-Scale Neural Audio Codec}, author={Siuzdak, Hubert}, year={2024}, url={https://github.com/hubertsiuzdak/snac}}
```
LoRA adapter over maya-research/maya1 (SNAC-24kHz). Cite the base + SNAC codec.
