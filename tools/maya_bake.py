"""Maya1 Slovenian TTS: zapeče stavke iz phrases.json (en del: SHARD od SHARDS) v audio/<hash>.mp3"""
import os, sys, time, json, subprocess, torch, soundfile as sf
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
done_f = f'audio/maya/shard_{SH:02d}.txt'
os.makedirs('audio/maya', exist_ok=True)
done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
ONLY = os.environ.get('ONLY', '').strip()
ALL = json.load(open('phrases.json', encoding='utf-8'))
if ONLY:
    want = set(json.load(open(ONLY, encoding='utf-8')))
    phrases = [p for p in ALL if p['t'] in want]
else:
    phrases = [p for i, p in enumerate(ALL) if i % N == SH and p['h'] not in done]
print(f'shard {SH}/{N}: {len(phrases)} stavkov', flush=True)
if not phrases: sys.exit(0)

tok = AutoTokenizer.from_pretrained("maya-research/maya1")
model = AutoModelForCausalLM.from_pretrained("maya-research/maya1", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "texdata/Maya1-Slovenian-TTS").merge_and_unload().eval()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
be = EspeakBackend("sl", with_stress=True, preserve_punctuation=True)

NAGL = json.load(open('tools/naglasi.json', encoding='utf-8')) if os.path.exists('tools/naglasi.json') else []
ACC = 'áéíóúàèìòùâêîôûŕ'
VOW_T = 'aeiouáéíóúàèìòùâêîôûə'
VOW_I = 'aeiouɛɔəɐæɪʊɑ'
def restress(tok, naglas):
    """premakne naglas v espeakovem zapisu besede na zlog, ki je v 'naglas' označen s strešico/ostrivcem"""
    word = naglas.lower(); n = None; k = 0
    for c in word:
        if c in VOW_T:
            if c in ACC: n = k; break
            k += 1
    if n is None: return tok
    t = tok.replace('ˈ', '').replace('ˌ', '')
    k = 0
    for i, c in enumerate(t):
        if c in VOW_I:
            if k == n:
                j = i
                if j > 0 and t[j-1] not in VOW_I + 'ː': j -= 1
                return t[:j] + 'ˈ' + t[j:]
            k += 1
            while i + 1 < len(t) and t[i+1] in 'ː': i += 1
    return tok
def phon(text):
    ph = phonemize_sl(text, be)
    rules = [r for r in NAGL if not r.get('stavek') or r['stavek'] == text]
    if not rules: return ph
    words = text.split(); toks = ph.split(' ')
    if len(words) == len(toks):
        for r in rules:
            for i, w in enumerate(words):
                if w.strip('.,!?:;').lower() == r['beseda'].lower():
                    toks[i] = r['ipa'] if r.get('ipa') else restress(toks[i], r['naglas'])
        new = ' '.join(toks)
        with open('audio/maya/fonemi.txt', 'a', encoding='utf-8') as f: f.write(f'{text}\t{ph}\t{new}\n')
        ph = new
    return ph
def synth(text):
    ph = phon(text)
    prompt = tok.apply_chat_template([{"role": "user", "content": f'<description="A Slovenian voice, speaker {SPK}."> {ph}'}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    mx = min(1800, 240 + len(text) * 12)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=mx, do_sample=True, temperature=0.75, top_p=0.9, repetition_penalty=1.1, eos_token_id=CODE_END, pad_token_id=CODE_END)
    gen = out[0, ids.shape[1]:].tolist()
    ended = CODE_END in gen
    s = [t for t in gen if SNAC_MIN <= t <= SNAC_MAX]
    if len(s) < 14: return None, False
    codes = [torch.tensor(c).unsqueeze(0) for c in unpack_snac(s)]
    with torch.inference_mode(): a = snac.decoder(snac.quantizer.from_codes(codes))[0, 0].float().numpy()
    return a, ended

def git_sync(msg):
    subprocess.run(['git', 'add', 'audio'], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 0: return
    subprocess.run(['git', 'commit', '-q', '-m', msg], check=True)
    for k in range(8):
        if subprocess.run(['git', 'pull', '-q', '--rebase', '-X', 'theirs', 'origin', 'main']).returncode == 0 and \
           subprocess.run(['git', 'push', '-q', 'origin', 'HEAD:main']).returncode == 0: return
        time.sleep(5 + k * 7)
    print('push failed', flush=True)

t0 = time.time(); n = 0
for p in phrases:
    exp = 0.4 + len(p['t']) * 0.085
    for attempt in range(3):
        a, ended = synth(p['t'])
        if a is not None and ended and 0.35 * exp < len(a) / 24000 < 2.6 * exp: break
        print('ponovi', p['t'], None if a is None else round(len(a) / 24000, 2), ended, flush=True); a = None
    if a is not None:
        sf.write('/tmp/x.wav', a, 24000)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', '/tmp/x.wav', '-af', 'silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,adelay=60,apad=pad_dur=0.12,loudnorm=I=-18:TP=-2',
                        '-ac', '1', '-ar', '24000', '-b:a', '48k', f"audio/{p['h']}.mp3"], check=True)
        with open(done_f, 'a') as f: f.write(p['h'] + '\n')
        n += 1
    if n and n % 12 == 0: git_sync(f'Glas Maya1: del {SH}, {n}/{len(phrases)}')
    print(f'{n}/{len(phrases)} {round(time.time()-t0)}s', flush=True)
git_sync(f'Glas Maya1: del {SH} končan ({n})')
