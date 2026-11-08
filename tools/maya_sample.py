import os, sys, time, json, torch, soundfile as sf, subprocess
sys.path.insert(0,'tools/maya')
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from snac import SNAC
from phonemizer.backend import EspeakBackend
from maya_tts import phonemize_sl, unpack_snac, CODE_END, SNAC_MIN, SNAC_MAX
torch.set_num_threads(os.cpu_count())
DT=getattr(torch, os.environ.get('DTYPE','bfloat16'))
t0=time.time()
tok=AutoTokenizer.from_pretrained("maya-research/maya1")
model=AutoModelForCausalLM.from_pretrained("maya-research/maya1", dtype=DT)
model=PeftModel.from_pretrained(model,"texdata/Maya1-Slovenian-TTS").merge_and_unload().eval()
snac=SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
be=EspeakBackend("sl", with_stress=True, preserve_punctuation=True)
log={'load_s':round(time.time()-t0,1),'cpus':os.cpu_count(),'dtype':str(DT),'items':[]}
TEXTS=["Ljubljana.","Ljubljana je glavno mesto Slovenije.","Kje je rjavi medved?","Soška postrv lahko zraste več kot en meter!","Ime Prekmurje pomeni dežela preko reke Mure.","Bravo!","Poleti in poberi zvezdice!"]
os.makedirs('tools/maya/vzorci',exist_ok=True)
for spk in os.environ.get('SPEAKERS','Artur-B-G0911').split(','):
  for i,text in enumerate(TEXTS):
    t1=time.time()
    ph=phonemize_sl(text,be); content=f'<description="A Slovenian voice, speaker {spk}."> {ph}'
    prompt=tok.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
    ids=tok(prompt,add_special_tokens=False,return_tensors="pt").input_ids
    with torch.inference_mode():
      out=model.generate(ids,max_new_tokens=1200,do_sample=True,temperature=0.75,top_p=0.9,repetition_penalty=1.1,eos_token_id=CODE_END,pad_token_id=CODE_END)
    gen=out[0,ids.shape[1]:].tolist(); s=[t for t in gen if SNAC_MIN<=t<=SNAC_MAX]
    codes=[torch.tensor(c).unsqueeze(0) for c in unpack_snac(s)]
    with torch.inference_mode(): audio=snac.decoder(snac.quantizer.from_codes(codes))[0,0].float().numpy()
    fn=f'tools/maya/vzorci/{spk}_{i}.wav'; sf.write(fn,audio,24000)
    log['items'].append({'spk':spk,'text':text,'tokens':len(gen),'sec':round(len(audio)/24000,2),'time':round(time.time()-t1,1)})
    print(log['items'][-1],flush=True); json.dump(log,open('tools/maya/vzorci/log.json','w'),ensure_ascii=False,indent=1)
