"""Zapeče slovenski glas: vsak stavek iz phrases.json -> audio/<hash>.mp3
Model: Piper sl_SI 'artur' (studijski posnetki korpusa ARTUR), z Hugging Face: speaches-ai/piper-sl_SI-artur-medium
"""
import json, os, sys, glob, wave, subprocess, tempfile
from huggingface_hub import snapshot_download

REPO = os.environ.get('MODEL_REPO', 'speaches-ai/piper-sl_SI-artur-medium')
LEN = float(os.environ.get('LENGTH_SCALE', '1.12'))   # >1 = počasneje (za otroke)
FORCE = os.environ.get('FORCE', '') == '1'

d = snapshot_download(REPO)
onnx = [f for f in glob.glob(d + '/**/*.onnx', recursive=True)][0]
cfg = [f for f in glob.glob(d + '/**/*.json', recursive=True) if f.endswith('.onnx.json') or os.path.basename(f) == 'config.json'][0]
print('model', onnx, cfg, flush=True)

from piper import PiperVoice
voice = PiperVoice.load(onnx, config_path=cfg)

def synth(text, path):
    try:  # piper-tts >= 1.3
        from piper import SynthesisConfig
        with wave.open(path, 'wb') as w: voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=LEN))
    except ImportError:
        with wave.open(path, 'wb') as w: voice.synthesize(text, w, length_scale=LEN)

phrases = json.load(open('phrases.json', encoding='utf-8'))
os.makedirs('audio', exist_ok=True)
todo = [p for p in phrases if FORCE or not os.path.exists(f"audio/{p['h']}.mp3")]
print(f'Stavkov: {len(phrases)}, za posneti: {len(todo)}', flush=True)
tmp = tempfile.mkdtemp()
for i, p in enumerate(todo):
    wav = os.path.join(tmp, 'x.wav'); synth(p['t'], wav)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', wav, '-af', 'adelay=60,apad=pad_dur=0.08', '-ac', '1', '-ar', '22050', '-b:a', '48k', f"audio/{p['h']}.mp3"], check=True)
    if (i + 1) % 200 == 0: print(f'{i+1}/{len(todo)}', flush=True)
files = sorted(f[:-4] for f in os.listdir('audio') if f.endswith('.mp3'))
json.dump(files, open('audio/index.json', 'w'))
print('Končano', len(todo), flush=True)
