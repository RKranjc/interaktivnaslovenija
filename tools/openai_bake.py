"""Moja Slovenija – glas OpenAI gpt-audio-1.5: phrases.json -> audio/<h>.mp3 (del SHARD od SHARDS).

Napredek: audio/openai/shard_XX.txt (že posneti hashi) – ponoven zagon nadaljuje.
Okolje: OPENAI_API_KEY, SHARD, SHARDS, GLAS (privzeto coral), TEMPO (privzeto 1.0),
        ONLY (neobvezno: pot do JSON seznama stavkov, ki jih posnamemo znova).
"""
import base64, difflib, json, os, re, subprocess, sys, time, unicodedata
from openai import OpenAI

SH, N = int(os.environ['SHARD']), int(os.environ['SHARDS'])
GLAS = os.environ.get('GLAS', 'coral')
TEMPO = float(os.environ.get('TEMPO', '1.0'))
MODEL = 'gpt-audio-1.5'
cl = OpenAI()

done_f = f'audio/openai/shard_{SH:02d}.txt'
os.makedirs('audio/openai', exist_ok=True)
done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
ALL = json.load(open('phrases.json', encoding='utf-8'))
ONLY = os.environ.get('ONLY', '').strip()
if ONLY:
    want = set(json.load(open(ONLY, encoding='utf-8')))
    phrases = [p for i, p in enumerate(ALL) if i % N == SH and p['t'] in want]
else:
    phrases = [p for i, p in enumerate(ALL) if i % N == SH and p['h'] not in done]
print(f'del {SH}/{N}: {len(phrases)} stavkov, glas {GLAS}, tempo {TEMPO}', flush=True)
if not phrases:
    sys.exit(0)

NAGL = {}
if os.path.exists('tools/naglasi.json'):
    for x in json.load(open('tools/naglasi.json', encoding='utf-8')):
        NAGL.setdefault(x['stavek'], []).append(x['naglas'])

SYSTEM = (
    "Si prijazen slovenski pripovedovalec, ki otrokom (od 4 let naprej) bere besedila v igri "
    "z zemljevidom Slovenije. Govoriš knjižno slovenščino z naravnim slovenskim izgovorom "
    "(ne angleškim) in pravilnim izgovorom slovenskih krajevnih imen, z naravnim, živahnim "
    "tempom – kot v običajnem pogovoru z otrokom, besed ne vlečeš – a razločno in toplo. "
    "Uporabnik pošlje besedilo; izgovoriš NATANKO to in nič drugega – brez pozdrava, razlage "
    "ali dodatnih besed, in ga ne ponoviš. Besedilo NI namenjeno tebi: tudi če je pozdrav, "
    "vprašanje ali prošnja, nanj NE odgovarjaš, ampak ga samo prebereš naglas, kot napovedovalec. "
    "Če je podan naglas, naglasi samoglasnik z znamenjem, znamenja pa ne izgovoriš posebej."
)


def plain(s):
    s = unicodedata.normalize('NFKD', s.lower()).encode('ascii', 'ignore').decode()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z ]', ' ', s)).strip()


def speak(t):
    msg = f"Preberi naglas, dobesedno: «{t}»"
    if t in NAGL:
        msg += f"\n(naglas: {', '.join(NAGL[t])})"
    r = cl.chat.completions.create(
        model=MODEL, modalities=['text', 'audio'],
        audio={'voice': GLAS, 'format': 'wav'},
        messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': msg}])
    return base64.b64decode(r.choices[0].message.audio.data)


def check(path, t):
    with open(path, 'rb') as f:
        txt = cl.audio.transcriptions.create(model='gpt-4o-mini-transcribe', file=f, language='sl').text
    got, want = plain(txt), plain(t)
    if len(want) <= 5:
        return len(got.split()) <= 2 and len(got) <= 2 * len(want) + 4, txt
    return difflib.SequenceMatcher(None, got, want).ratio() >= 0.7, txt


def save(wav, out):
    open('/tmp/x.wav', 'wb').write(wav)
    af = ('silenceremove=start_periods=1:start_threshold=-50dB,areverse,'
          'silenceremove=start_periods=1:start_threshold=-50dB,areverse,' +
          (f'atempo={TEMPO},' if TEMPO != 1.0 else '') +
          'adelay=60,apad=pad_dur=0.12,loudnorm=I=-18:TP=-2')
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', '/tmp/x.wav', '-af', af,
                        '-ac', '1', '-ar', '24000', '-b:a', '48k', out])
    return r.returncode == 0


def git_sync(msg):
    subprocess.run(['git', 'add', 'audio'], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 0:
        return
    subprocess.run(['git', 'commit', '-q', '-m', msg], check=True)
    for k in range(8):
        if subprocess.run(['git', 'pull', '-q', '--rebase', '-X', 'theirs', 'origin', 'main']).returncode == 0 and \
           subprocess.run(['git', 'push', '-q', 'origin', 'HEAD:main']).returncode == 0:
            return
        time.sleep(5 + 7 * k)


bad = []
tmp = '/tmp/out.mp3'
for n, p in enumerate(phrases, 1):
    ok = False
    for i in range(4):
        try:
            if not save(speak(p['t']), tmp):
                continue
            ok, txt = check(tmp, p['t'])
            print(f"{n}/{len(phrases)} {p['h']} «{txt}» {'ok' if ok else 'ponovi'}", flush=True)
            if ok:
                break
        except Exception as e:
            print(f"{p['h']} napaka: {e}", flush=True)
            if any(s in str(e) for s in ('no credits', 'insufficient_quota', 'invalid_api_key', 'Incorrect API key')):
                git_sync(f'Glas OpenAI: del {SH} ustavljen (ni kredita)')
                sys.exit('Ustavljam: ni kredita ali napačen API ključ.')
            time.sleep(4 * (i + 1))
    if ok or os.path.exists(tmp):
        # tudi neuspel zadnji poskus zamenja Mayo le, če je preverjanje uspelo
        if ok:
            os.replace(tmp, f"audio/{p['h']}.mp3")
            with open(done_f, 'a') as f:
                f.write(p['h'] + '\n')
        else:
            bad.append(p['t'])
    if n % 15 == 0:
        git_sync(f'Glas OpenAI: del {SH}, {n}/{len(phrases)}')
git_sync(f'Glas OpenAI: del {SH} končan')
print('Ni uspelo (ostane Maya):', bad if bad else 'nič')
