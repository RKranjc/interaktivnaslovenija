"""Posname slovenske stavke iz phrases.json z glasom Azure (sl-SI-PetraNeural) v audio/<hash>.mp3."""
import json, os, sys, time, html, urllib.request, urllib.error, subprocess

KEY = os.environ.get('AZURE_SPEECH_KEY', '').strip()
REGION = os.environ.get('AZURE_SPEECH_REGION', 'westeurope').strip()
VOICE = os.environ.get('VOICE', 'sl-SI-PetraNeural')
RATE = os.environ.get('RATE', '-6%')
if not KEY:
    sys.exit('Manjka skrivnost AZURE_SPEECH_KEY (Settings → Secrets and variables → Actions).')

phrases = json.load(open('phrases.json', encoding='utf-8'))
os.makedirs('audio', exist_ok=True)
url = f'https://{REGION}.tts.speech.microsoft.com/cognitiveservices/v1'
todo = [p for p in phrases if not os.path.exists(f"audio/{p['h']}.mp3")]
print(f'Stavkov: {len(phrases)}, za posneti: {len(todo)}', flush=True)

def commit(msg):
    files = sorted(f[:-4] for f in os.listdir('audio') if f.endswith('.mp3'))
    json.dump(files, open('audio/index.json', 'w'))
    subprocess.run(['git', 'add', 'audio'], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode != 0:
        subprocess.run(['git', 'commit', '-q', '-m', msg], check=True)
        subprocess.run(['git', 'pull', '-q', '--rebase', 'origin', 'main'], check=True)
        subprocess.run(['git', 'push', '-q', 'origin', 'HEAD:main'], check=True)

done = 0
for i, p in enumerate(todo):
    ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='sl-SI'>"
            f"<voice name='{VOICE}'><prosody rate='{RATE}'>{html.escape(p['t'])}</prosody></voice></speak>")
    for attempt in range(8):
        req = urllib.request.Request(url, data=ssml.encode('utf-8'), method='POST', headers={
            'Ocp-Apim-Subscription-Key': KEY,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-24khz-48kbitrate-mono-mp3',
            'User-Agent': 'moja-slovenija-3d'})
        try:
            data = urllib.request.urlopen(req, timeout=60).read()
            open(f"audio/{p['h']}.mp3", 'wb').write(data); done += 1
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(int(e.headers.get('Retry-After') or 15)); continue
            sys.exit(f'Napaka {e.code}: {e.read()[:300]}')
        except Exception as e:
            time.sleep(10)
    time.sleep(float(os.environ.get('PAUSE', '3.1')))   # brezplačni paket: ~20 zahtev na minuto
    if (i + 1) % 150 == 0:
        print(f'{i+1}/{len(todo)}', flush=True); commit(f'Glas: {i+1}/{len(todo)} stavkov')
commit(f'Glas: posnetih {done} novih stavkov')
print('Končano', done, flush=True)
