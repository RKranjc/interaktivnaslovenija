"""Maya1 slovenski glas za igro Zlogi: zlogi/fraze.json -> zlogi/zvok/<k>.mp3 (del SHARD od SHARDS)"""
import os, sys, json, subprocess, torch, soundfile as sf
sys.path.insert(0, 'tools/maya')
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from snac import SNAC
from phonemizer.backend import EspeakBackend
from maya_tts import phonemize_sl, unpack_snac, CODE_END, SNAC_MIN, SNAC_MAX

SH, N = int(os.environ['SHARD']), int(os.environ['SHARDS'])
SPK = os.environ.get('SPEAKER', 'Artur-B-G0911')
torch.set_num_threads(os.cpu_count())
os.makedirs('zlogi/zvok', exist_ok=True)
ALL = json.load(open('zlogi/fraze.json', encoding='utf-8'))
todo = [p for i, p in enumerate(ALL) if i % N == SH and not os.path.exists(f"zlogi/zvok/{p['k']}.mp3")]
print(f'del {SH}/{N}: {len(todo)}', flush=True)
if not todo: sys.exit(0)

tok = AutoTokenizer.from_pretrained("maya-research/maya1")
model = AutoModelForCausalLM.from_pretrained("maya-research/maya1", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "texdata/Maya1-Slovenian-TTS").merge_and_unload().eval()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
be = EspeakBackend("sl", with_stress=True, preserve_punctuation=True)

ACC = 'áéíóúàèìòùâêîôûŕ'; VOW_T = 'aeiouáéíóúàèìòùâêîôûə'; VOW_I = 'aeiouɛɔəɐæɪʊɑ'
def restress(t, naglas):
    """'ˈ' pred naglašeni samoglasnik (kot v tools/maya_bake.py)"""
    n = None; k = 0; short = False
    for c in naglas.lower():
        if c in VOW_T:
            if c in ACC: n = k; short = c in 'àèìòù'; break
            k += 1
    if n is None: return t
    t = t.replace('ˈ', '').replace('ˌ', '').replace('ː', ''); k = 0
    for i, c in enumerate(t):
        if c in VOW_I:
            if k == n: return t[:i] + 'ˈ' + c + ('' if short else 'ː') + t[i+1:]
            k += 1
    return t
def phon(p):
    ph = phonemize_sl(p['t'], be)
    if p.get('n'):
        body = ph.rstrip('.!?'); ph = restress(body, p['n']) + ph[len(body):]
    print(p['t'], ph, flush=True)
    return ph
def synth(p):
    prompt = tok.apply_chat_template([{"role": "user", "content": f'<description="A Slovenian voice, speaker {SPK}."> {phon(p)}'}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=min(1800, 240 + len(p['t']) * 12), do_sample=True, temperature=0.75, top_p=0.9,
                             repetition_penalty=1.1, eos_token_id=CODE_END, pad_token_id=CODE_END)
    gen = out[0, ids.shape[1]:].tolist(); s = [t for t in gen if SNAC_MIN <= t <= SNAC_MAX]
    if len(s) < 14: return None, False
    codes = [torch.tensor(c).unsqueeze(0) for c in unpack_snac(s)]
    with torch.inference_mode(): a = snac.decoder(snac.quantizer.from_codes(codes))[0, 0].float().numpy()
    return a, CODE_END in gen

import numpy as np
def save(a, out):
    sf.write('/tmp/x.wav', np.nan_to_num(a).clip(-1, 1), 24000)
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', '/tmp/x.wav', '-af', 'silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,adelay=40,apad=pad_dur=0.08,loudnorm=I=-18:TP=-2',
                        '-ac', '1', '-ar', '24000', '-b:a', '48k', out])
    return r.returncode == 0

for n, p in enumerate(todo):
    exp = 0.4 + len(p['t']) * 0.085; ok = False
    for attempt in range(5):
        a, ended = synth(p)
        if a is None or not ended or not 0.35 * exp < len(a) / 24000 < 2.6 * exp:
            print('ponovi', p['t'], None if a is None else round(len(a) / 24000, 2), ended, flush=True); continue
        if save(a, f"zlogi/zvok/{p['k']}.mp3"): ok = True; break
        print('ffmpeg ni uspel, ponovi', p['t'], flush=True)
    print(f"{n+1}/{len(todo)}", 'ok' if ok else 'PRESKOČENO', flush=True)
