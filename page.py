PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic Lab Vision</title>
<style>
:root{--bg:#0f1115;--fg:#f4f6fa;--mut:#aab2c0;--card:#1a1e26;--ok:#38d996;--warn:#ffc83d;--bad:#ff6b6b;--acc:#5aa9ff}
body{margin:0;background:var(--bg);color:var(--fg);font:20px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:32px;margin:0 0 4px}p.s{color:var(--mut);margin:0 0 20px}
.card{background:var(--card);border-radius:14px;padding:18px;margin:16px 0}
input,button{font:inherit;border-radius:10px;border:2px solid #39414f;background:#0f1115;color:var(--fg);padding:12px 14px}
button{background:var(--acc);color:#07101c;border:0;font-weight:700;cursor:pointer}
button:disabled{opacity:.5}
button.alt{background:#2a3140;color:var(--fg)}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
#v{font-size:26px;font-weight:800;padding:12px 16px;border-radius:12px;display:none}
.ESTIMATE{background:#0f3a2b;color:var(--ok)}
.NEEDS_HUMAN{background:#3d3210;color:var(--warn)}
.RETAKE{background:#401a1a;color:var(--bad)}
img{max-width:100%;border-radius:10px}
ol{padding-left:22px}
li{margin:8px 0}
code{color:var(--acc)}
.m{color:var(--mut);font-size:17px}
</style></head><body><main>
<h1>Agentic Lab Vision</h1>
<p class="s">Photo of a culture plate in. OpenCV 5 checks the photo, counts colonies, and decides what happens next: repair the photo, ask for a retake, or hand you an estimate to approve.</p>
<div class="card"><div class="row">
<input id="f" type="file" accept="image/*" aria-label="Plate photo">
<input id="k" type="password" placeholder="API key" aria-label="API key" size="18">
<button id="go">Analyse</button></div>
<p class="m" id="st">Choose a photo of a petri dish. It is resized in your browser before upload.</p></div>
<div id="out" class="card" style="display:none">
<div id="v"></div><p id="rs"></p>
<img id="ov" alt="Plate with detected colonies circled in red">
<div class="row" id="hd" style="display:none;margin-top:12px">
<button id="ap">Approve this estimate</button><button class="alt" id="rt">Request retake</button><span class="m" id="hm"></span></div>
<h3>What the agent saw and did</h3><ol id="tr"></ol></div>
<script>
const $=i=>document.getElementById(i);let last=null;
async function shrink(file,max){max=max||1600;const bm=await createImageBitmap(file);const s=Math.min(1,max/Math.max(bm.width,bm.height));
const c=document.createElement('canvas');c.width=Math.round(bm.width*s);c.height=Math.round(bm.height*s);c.getContext('2d').drawImage(bm,0,0,c.width,c.height);
return new Promise(r=>c.toBlob(b=>{const fr=new FileReader();fr.onload=()=>r(fr.result.split(',')[1]);fr.readAsDataURL(b)},'image/jpeg',.9))}
async function call(path,body){const r=await fetch(path,{method:'POST',headers:{'content-type':'application/json','x-api-key':$('k').value},body:JSON.stringify(body)});
const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);return j}
function esc(t){return String(t).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}
$('go').onclick=async()=>{const f=$('f').files[0];if(!f){$('st').textContent='Pick a photo first.';return}
$('go').disabled=true;$('st').textContent='Analysing...';
try{const j=await call('/analyze',{image:await shrink(f)});last=j;
$('out').style.display='block';const v=$('v');v.style.display='block';v.className=j.outcome;
v.textContent=j.outcome==='ESTIMATE'?('ESTIMATE: about '+j.count+' colonies'):j.outcome==='RETAKE'?'RETAKE REQUESTED':('HUMAN REVIEW: '+(j.count==null?'no':j.count)+' colonies counted');
$('rs').textContent=j.reason+' ('+j.ms+' ms, OpenCV '+j.opencv+')';$('ov').src='data:image/jpeg;base64,'+j.overlay;
$('hd').style.display=(j.outcome==='NEEDS_HUMAN'||j.outcome==='ESTIMATE')?'flex':'none';$('hm').textContent='';
$('tr').innerHTML=j.trace.map(t=>'<li><b>Step '+t.step+'</b>: dish '+(t.observation.dish_found?'found':'NOT found')+', sharpness '+t.observation.sharpness+', brightness '+t.observation.brightness+', glare '+t.observation.glare+', coverage '+t.observation.coverage+', blobs '+t.observation.count+'<br><code>'+esc(t.decision)+'</code></li>').join('');
$('st').textContent='Done.'}catch(e){$('st').textContent='Error: '+e.message}$('go').disabled=false};
[['ap','approve'],['rt','retake']].forEach(p=>{$(p[0]).onclick=async()=>{try{const r=await call('/decision',{id:last.id,decision:p[1]});$('hm').textContent='Recorded: '+r.recorded}catch(e){$('hm').textContent=e.message}}});
</script></main></body></html>"""
