"""Zlogolov – glas OpenAI gpt-audio-1.5: zlogi/fraze.json -> mp3.

Nastavitve v tools/zlogi_openai.json:
  {"nacin": "test" | "vse", "glas": "marin", "glasovi_test": [...], "znova": false}
  test: 10 zlogov + 5 besed + 2 pohvali za vsak glas -> zlogi/zvok_test/<glas>/<k>.mp3
  vse:  manjkajoče (ali vse, če znova=true) -> zlogi/zvok/<k>.mp3
Vsak posnetek preveri s prepisom; če model doda besede, posname znova.
"""
import base64, json, os, re, subprocess, sys, time, unicodedata
from openai import OpenAI

CFG = json.load(open('tools/zlogi_openai.json', encoding='utf-8'))
ALL = json.load(open('zlogi/fraze.json', encoding='utf-8'))
MODEL = 'gpt-audio-1.5'
cl = OpenAI()

SYSTEM = (
    "Si prijazen slovenski učitelj, ki otrokom (6–7 let) bere zloge in besede. "
    "Govoriš knjižno slovenščino z naravnim slovenskim izgovorom (ne angleškim), "
    "počasi, razločno in toplo. Uporabnik pošlje besedilo; izgovoriš NATANKO to in "
    "nič drugega – brez pozdrava, razlage ali dodatnih besed, in ga ne ponoviš. "
    "Zlog (npr. »va«) izgovoriš kot zlog [va], ne črkuješ ga. "
    "Če je podan naglas, naglasi samoglasnik z znamenjem (á, é, í, ó, ú), "
    "znamenja pa ne izgovoriš posebej."
)


def plain(s):
    s = unicodedata.normalize('NFKD', s.lower()).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z ]', '', s).strip()


def speak(p, glas):
    msg = p['t'].rstrip('.')
    if p.get('n'):
        msg += f"\n(naglas: {p['n']})"
    r = cl.chat.completions.create(
        model=MODEL, modalities=['text', 'audio'],
        audio={'voice': glas, 'format': 'wav'},
        messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': msg}])
    return base64.b64decode(r.choices[0].message.audio.data)


def check(path, p):
    with open(path, 'rb') as f:
        txt = cl.audio.transcriptions.create(model='gpt-4o-mini-transcribe', file=f, language='sl').text
    got, want = plain(txt), plain(p['t'])
    ok = len(got.split()) <= len(want.split()) + 1 and len(got) <= 2 * len(want) + 4
    return ok, txt


def save(wav, out):
    open('/tmp/x.wav', 'wb').write(wav)
    # enako kot pri Mayi: obreži tišino, normaliziraj glasnost, mono 24 kHz
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', '/tmp/x.wav', '-af',
                        'silenceremove=start_periods=1:start_threshold=-50dB,areverse,'
                        'silenceremove=start_periods=1:start_threshold=-50dB,areverse,'
                        'adelay=40,apad=pad_dur=0.08,loudnorm=I=-18:TP=-2',
                        '-ac', '1', '-ar', '24000', '-b:a', '48k', out])
    return r.returncode == 0


def record(p, glas, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for i in range(4):
        try:
            if not save(speak(p, glas), out):
                continue
            ok, txt = check(out, p)
            print(f"{p['k']:14} {glas:7} «{txt}» {'ok' if ok else 'ponovi'}", flush=True)
            if ok:
                return True
        except Exception as e:
            print(f"{p['k']} napaka: {e}", flush=True)
            time.sleep(4 * (i + 1))
    print(f"{p['k']} PREVERI ROČNO (obdržan zadnji poskus)", flush=True)
    return False


bad = []
if CFG['nacin'] == 'test':
    by = {p['k']: p for p in ALL}
    pick = [by[k] for k in CFG['test_kljuci']]
    pick += [{'k': f's-{i + 1}', 't': t} for i, t in enumerate(CFG.get('test_stavki', []))]
    for g in CFG['glasovi_test']:
        for p in pick:
            out = f"zlogi/zvok_test/{g}/{p['k']}.mp3"
            if CFG.get('znova') or not os.path.exists(out):
                if not record(p, g, out):
                    bad.append(f"{g}/{p['k']}")
    json.dump({'glasovi': CFG['glasovi_test'], 'fraze': [[p['k'], p['t'].rstrip('.')] for p in pick]},
              open('zlogi/zvok_test/seznam.json', 'w', encoding='utf-8'), ensure_ascii=False)
else:
    for p in ALL:
        out = f"zlogi/zvok/{p['k']}.mp3"
        if CFG.get('znova') or not os.path.exists(out):
            if not record(p, CFG['glas'], out):
                bad.append(p['k'])
print('Preveri ročno:', ', '.join(bad) if bad else 'nič')
