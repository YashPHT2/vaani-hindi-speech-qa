"""Create a local, network-free listening form; never fabricates annotations."""
import hashlib
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'artifacts/recoverability-targets-reviewed.jsonl'
    targets=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    manifest={r['id']:r for r in map(json.loads,(HERE.parent/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    cards=[]
    payload=[]
    for index,row in enumerate(targets):
        original=manifest[row['id']]
        audio=HERE.parent/'data'/original['audio']
        assert audio.exists() and hashlib.sha256(audio.read_bytes()).hexdigest()==original['audio_sha256']
        e=html.escape
        cards.append(f'''<section data-index="{index}">
<h2>{e(row['source_id'])} · {e(row['condition'])}</h2>
<audio controls preload="none" src="../data/{e(original['audio'])}"></audio>
<p><b>Question:</b> {e(row['question'])}</p>
<p><b>Publisher target (unverified):</b> {e(row['answer_critical_reference_phrase'])}</p>
<details><summary>Publisher sentence and ASR output (reference-guided review)</summary>
<p>{e(row['publisher_reference_text'])}</p><p><b>ASR:</b> {e(row['current_asr_full_text'])}</p></details>
<label><input type="checkbox" name="listened"> I listened to this recording.</label>
<label>Is the publisher target actually spoken and intelligible?
<select name="verdict"><option value="">Unreviewed</option><option value="audible">Clearly audible</option><option value="not_spoken">Not spoken / reference mismatch</option><option value="uncertain">Uncertain / unintelligible</option></select></label>
<label>Words you actually hear near the target <input name="heard" type="text" placeholder="Leave blank if uncertain" lang="hi"></label>
<label>Approximate target interval in seconds (optional; do not guess)
<input name="start" type="number" min="0" max="{original['duration_s']}" step="0.01" placeholder="Start">
<input name="end" type="number" min="0" max="{original['duration_s']}" step="0.01" placeholder="End"></label>
<label>Meaning of target retained in the displayed ASR?
<select name="retained"><option value="unknown">Uncertain / not assessed</option><option value="yes">Yes, including recognizable spelling</option><option value="no">No</option></select></label>
<label>Notes <textarea name="notes" rows="2"></textarea></label>
</section>''')
        payload.append(dict(id=row['id'],source_id=row['source_id'],condition=row['condition'],
            audio_sha256=original['audio_sha256'],duration_s=original['duration_s'],
            publisher_target=row['answer_critical_reference_phrase']))
    data=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recoverability listening review</title><style>
body{font:17px/1.55 system-ui,sans-serif;max-width:940px;margin:32px auto;padding:0 20px;background:#f5f6f8;color:#182338}
section{background:white;padding:24px;border:1px solid #cad2de;border-radius:12px;margin:22px 0}h1{line-height:1.2}h2{font-size:20px}
label{display:block;margin:14px 0}input[type=text],textarea{display:block;width:95%;font:inherit;padding:8px}select{display:block;font:inherit;max-width:100%;padding:6px}
audio{width:100%}input[type=number]{width:120px;padding:7px}button{font:inherit;padding:12px 18px;cursor:pointer;background:#244f93;color:white;border:0;border-radius:7px}
.note{background:#e4edf9;padding:16px;border-radius:8px}details{margin:12px 0}footer{padding:24px 0}</style>
<h1>Verify the spoken evidence</h1><p class="note">Ten views from five recordings. Clean and noisy views share a source; count each source once. This page runs locally and sends nothing to a server. Publisher text and prior ASR are visible, so this is reference-guided listening, not a blinded transcription study.</p>
<p>Listen before answering. Mark uncertain when needed. No timestamps or words have been prefilled as verified. An approximate interval is useful for later oracle-window work but is optional for a full-recording decoder diagnostic.</p>
'''+'\n'.join(cards)+'''
<footer><button id="save">Save listening annotations</button><p id="status" aria-live="polite"></p>
<p>Save/download <b>recoverability-listening-annotations.json</b>, then tell the research chat that the review is saved. Partial reviews are allowed and remain explicitly incomplete.</p></footer>
<script>
const clips='''+data+''';
const auditSha="'''+digest+'''";
document.getElementById('save').addEventListener('click',()=>{
  const annotations=[...document.querySelectorAll('section')].map((card,i)=>{
    const field=n=>card.querySelector(`[name="${n}"]`);
    const start=field('start').value===''?null:Number(field('start').value);
    const end=field('end').value===''?null:Number(field('end').value);
    if((start===null)!==(end===null) || start!==null && !(start>=0 && end>start && end<=clips[i].duration_s))
      throw new Error('Check start/end for '+clips[i].id);
    const listened=field('listened').checked;
    const verdict=field('verdict').value;
    return {...clips[i],human_listened:listened,verdict:verdict||'unreviewed',
      heard_phrase:field('heard').value.trim(),target_interval_s:start===null?null:[start,end],
      asr_semantic_target_retained:field('retained').value,notes:field('notes').value.trim(),
      target_spoken_verified:listened && verdict==='audible',verification_method:'reference_guided_human_listening'};
  });
  const result={schema:'recoverability-listening-v1',target_audit_sha256:auditSha,
    reviewer_type:'user_human_listener',created_at:new Date().toISOString(),annotations};
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+'\\n'],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='recoverability-listening-annotations.json';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
  document.getElementById('status').textContent='Saved '+annotations.filter(x=>x.human_listened).length+' listened views. Tell the chat when the file is saved.';
});
</script></html>'''
    (HERE/'recoverability_listening_review.html').write_text(page,encoding='utf-8')
    print('Local listening form created:10views,5sources; all annotation fields unreviewed.')


if __name__=='__main__':
    main()
