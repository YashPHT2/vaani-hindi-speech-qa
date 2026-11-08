"""Create a local targeted boundary/placebo listening form, no annotation inference."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    manifest=[json.loads(s) for s in (HERE.parent/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=sorted([r for r in manifest if r['source_id']=='fleurs-hi-1806'],key=lambda r:r['condition'])
    assert len(rows)==2
    clips=[dict(id=r['id'],condition=r['condition'],duration_s=r['duration_s'],audio_sha256=r['audio_sha256'],
        src='../data/'+r['audio']) for r in rows]
    for row in rows:
        assert hashlib.sha256((HERE.parent/'data'/row['audio']).read_bytes()).hexdigest()==row['audio_sha256']
    payload=json.dumps(clips,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phrase boundary review</title><style>
body{font:17px/1.6 system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 20px;background:#f5f7fa;color:#15243b}
section{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:22px;margin:18px 0}
label{display:block;margin:14px 0}input[type=number]{font:inherit;width:110px;padding:6px}input[type=text],textarea{font:inherit;width:96%;padding:7px}
button{font:inherit;padding:8px 13px;margin:5px;border:1px solid #a7b9d2;border-radius:6px;cursor:pointer}
audio{width:100%}.note{background:#e7eef9;padding:15px;border-radius:8px}.value{font-weight:600}#save{background:#214e91;color:white}#status{font-weight:600}</style>
<h1>Where exactly is प्रकाश प्रदूषण?</h1>
<p class="note">One recording with clean and noisy versions. The previous2–3second mark was approximate. Please refine it by listening; the form does not treat it as validated. No data is sent to a server.</p>
<section><h2>1. Locate the complete phrase</h2>
<p>Set the start just before प्रकाश and the end just after प्रदूषण. Use the context buttons to check for clipped sounds. Boundaries can be approximate; record your uncertainty rather than guessing precision.</p>
<label>Phrase start (seconds) <input id="start" type="number" min="0" max="13.68" step="0.01" value="2"></label>
<label>Phrase end (seconds) <input id="end" type="number" min="0" max="13.68" step="0.01" value="3"></label>
<label>Uncertainty on each boundary, ±seconds <input id="uncertainty" type="number" min="0.01" max="1" step="0.01" placeholder="e.g.0.10"></label>
<p>Planned target-mask window, including that margin: <span id="target" class="value">Enter boundary uncertainty.</span></p>
<div id="clips"></div>
<label><input id="verified" type="checkbox"> I listened and this interval contains the complete target phrase; I checked both versions.</label>
</section>
<section><h2>2. Check an unrelated interval</h2>
<p>The placebo mask must have the same duration and must not contain प्रकाश प्रदूषण. Start8seconds is only a proposal; listen and change it if necessary. Both masks will use the same replacement and edge fades.</p>
<label>Placebo start (seconds) <input id="placebo" type="number" min="0" max="13.68" step="0.01" value="8"></label>
<p>Placebo interval: <span id="placeboRange" class="value">Enter boundary uncertainty.</span></p>
<button data-play="placebo" data-index="0">Listen: clean placebo</button><button data-play="placebo" data-index="1">Listen: noisy placebo</button>
<label><input id="unrelated" type="checkbox"> I listened: the placebo interval is unrelated speech and does not contain the target phrase.</label>
</section>
<section><h2>3. Save</h2><label>Notes or uncertainty <textarea id="notes" rows="3"></textarea></label>
<button id="save">Save boundary review</button><p id="status" aria-live="polite"></p>
<p>Download <b>vaani-localization-boundaries.json</b>, then tell the research chat <b>“boundaries saved.”</b></p></section>
<script>
const clips='''+payload+''';
const number=id=>document.getElementById(id).value===''?NaN:Number(document.getElementById(id).value);
const box=document.getElementById('clips');
clips.forEach((clip,i)=>{const d=document.createElement('div');d.innerHTML=`<h3>${clip.condition}</h3><audio id="audio${i}" controls preload="metadata" src="${clip.src}"></audio><button data-play="phrase" data-index="${i}">Play phrase</button><button data-play="context" data-index="${i}">Play phrase with context</button><button data-play="target" data-index="${i}">Play planned target-mask window</button>`;box.appendChild(d);});
function ranges(){const s=number('start'),e=number('end'),u=number('uncertainty'),p=number('placebo');
 if(![s,e,u,p].every(Number.isFinite)||!(s>=0&&e>s&&e<=13.68&&u>0&&u<=1))return null;
 const target=[Math.max(0,s-u),Math.min(13.68,e+u)],placebo=[p,p+target[1]-target[0]];
 return {phrase:[s,e],uncertainty:u,target,placebo};}
function show(){const r=ranges();document.getElementById('target').textContent=r?`${r.target[0].toFixed(2)}–${r.target[1].toFixed(2)} seconds`:'Enter valid bounds and uncertainty.';
 document.getElementById('placeboRange').textContent=r?`${r.placebo[0].toFixed(2)}–${r.placebo[1].toFixed(2)} seconds`:'Enter valid bounds and uncertainty.';}
['start','end','uncertainty','placebo'].forEach(id=>document.getElementById(id).addEventListener('input',show));
const players=clips.map((_,i)=>document.getElementById('audio'+i));
players.forEach(a=>a.addEventListener('timeupdate',()=>{if(a.stopAt!==undefined&&a.currentTime>=a.stopAt){a.pause();a.stopAt=undefined;}}));
document.addEventListener('click',event=>{const button=event.target.closest('button[data-play]');if(!button)return;
 const type=button.dataset.play;const r=ranges();let range;
 if(type==='phrase'||type==='context'){const s=number('start'),e=number('end');if(!(Number.isFinite(s)&&Number.isFinite(e)&&s>=0&&e>s&&e<=13.68))return;
 range=type==='context'?[Math.max(0,s-.5),Math.min(13.68,e+.5)]:[s,e];}
 else{if(!r)return;range=r[type];}if(range[0]<0||range[1]>13.68)return;
 players.forEach(a=>a.pause());const audio=players[Number(button.dataset.index)];audio.currentTime=range[0];audio.stopAt=range[1];audio.play().catch(()=>{});});
document.getElementById('save').addEventListener('click',()=>{const status=document.getElementById('status');const r=ranges();
 if(!r||r.placebo[0]<0||r.placebo[1]>13.68||Math.max(r.placebo[0],r.target[0])<Math.min(r.placebo[1],r.target[1])){status.textContent='Check boundaries: intervals must fit the clip, have equal duration and not overlap.';return;}
 if(!document.getElementById('verified').checked||!document.getElementById('unrelated').checked){status.textContent='Listen and confirm the phrase and unrelated placebo interval before saving.';return;}
 const data={schema:'vaani-localization-boundaries-v1',source_id:'fleurs-hi-1806',target_phrase:'प्रकाश प्रदूषण',clips,
 phrase_interval_s:r.phrase,boundary_uncertainty_s:r.uncertainty,target_mask_interval_s:r.target,placebo_mask_interval_s:r.placebo,
 complete_phrase_checked_both_versions:true,placebo_checked_unrelated:true,method:'single-user reference-guided listening; approximate boundaries',
 notes:document.getElementById('notes').value,created_at:new Date().toISOString()};
 const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\\n'],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='vaani-localization-boundaries.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='Saved boundary review. Tell the research chat: boundaries saved.';
});show();
</script></html>'''
    (HERE/'localization_boundary_review.html').write_text(page,encoding='utf-8')
    print('Local boundary and equal-durationplacebo review created; annotations remain unvalidated.')


if __name__=='__main__':
    main()
