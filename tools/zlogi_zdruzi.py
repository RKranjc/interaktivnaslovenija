"""Združi zlogi/zvok/*.mp3 v zlogi/zvok.mp3 + zlogi/zvok.json ({ključ: [začetek, trajanje]}), 0,3 s premora."""
import glob, json, os, subprocess

GAP = 0.3
files = sorted(glob.glob('zlogi/zvok/*.mp3'))
idx, parts, t = {}, [], GAP
for f in files:
    # dejanska dekodirana dolžina (ffprobe zaradi mp3 polnila poroča predolgo)
    pcm = subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-i', f, '-f', 's16le',
                                   '-ac', '1', '-ar', '24000', '-'])
    d = len(pcm) / 2 / 24000
    idx[os.path.basename(f)[:-4]] = [round(t, 3), round(d, 3)]
    parts.append(f)
    t += d + GAP

# vsak posnetek dekodiraj, dodaj 0,3 s tišine, nato vse skupaj kodiraj enkrat
inputs, filt = [], []
for i, f in enumerate(parts):
    inputs += ['-i', f]
    filt.append(f'[{i}:a]aresample=24000,aformat=channel_layouts=mono,apad=pad_dur={GAP}[a{i}]')
lead = f'anullsrc=r=24000:cl=mono,atrim=0:{GAP}[s]'
concat = '[s]' + ''.join(f'[a{i}]' for i in range(len(parts))) + f'concat=n={len(parts) + 1}:v=0:a=1[o]'
subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', *inputs, '-filter_complex',
                ';'.join([lead] + filt + [concat]), '-map', '[o]', '-ac', '1', '-ar', '24000',
                '-b:a', '56k', 'zlogi/zvok.mp3'], check=True)
json.dump(idx, open('zlogi/zvok.json', 'w', encoding='utf-8'), separators=(',', ':'))
print(len(idx), 'posnetkov,', round(t, 1), 's')
