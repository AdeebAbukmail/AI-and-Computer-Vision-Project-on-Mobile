import os, logging, urllib.request
import cv2, numpy as np, mediapipe as mp
from flask import Flask, Response, request, jsonify
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.exists(MODEL):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", MODEL)

hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=MODEL), num_hands=2,
    min_hand_detection_confidence=0.6, min_tracking_confidence=0.6))

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

PAGE = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>تحكم بالأيدي</title><style>
:root{--g:#0a9d45}
*{box-sizing:border-box}
body{margin:0;background:#000;color:var(--g);font-family:sans-serif;display:flex;
flex-direction:column;align-items:center;gap:8px;padding:8px;min-height:100vh}
h3{margin:2px}
#box{position:relative;width:100%;max-width:900px;height:calc(100dvh - 250px);min-height:360px;
border:3px solid var(--g);border-radius:12px;overflow:hidden;background:#010a04;box-shadow:0 0 14px #0a9d4544}
#box.alarm{border-color:#e00;box-shadow:0 0 30px #e00}
video,canvas{position:absolute;inset:0;width:100%;height:100%}
video{object-fit:cover}
.row{display:flex;gap:8px;width:100%;max-width:900px}
button{flex:1;padding:14px;font-size:18px;border-radius:10px;border:2px solid #000;font-weight:bold}
#cam{background:#b8860b;color:#000}
#go{background:var(--g);color:#000}
#st{font-size:19px;text-align:center}
.i{font-size:16px;color:#7fbf8f}
</style></head><body>
<h3>🖐 تحكم بالأيدي</h3>
<div id="box"><video id="v" playsinline muted></video><canvas id="c"></canvas></div>
<div class="row"><button id="cam">الكاميرا: خلفية 🔄</button><button id="go">تشغيل</button></div>
<div id="st">اضغط تشغيل</div>
<div class="i" id="info">الأيدي المكتشفة: 0</div>
<div class="i" id="vol"></div>
<div class="i" id="tor">🔦 الكشاف: مطفأ</div>
<script>
const $=id=>document.getElementById(id), v=$('v'), c=$('c'), x=c.getContext('2d'), box=$('box');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const cap=document.createElement('canvas'), cx=cap.getContext('2d');
let stream=null, mode='environment', starting=false, target=[], cur=[], miss=0;
let last='', n=0, stable='none', vol=0.5, lastTick=0, torchOn=false, torchFail=false;
let actx, master, bellTm;
const L=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TXT={fist:'🔔 قبضة: الجرس شغال',one:'☝️ رفع الصوت',two:'✌️ الكشاف',open:'✋ يد مفتوحة',none:'ما في يد'};

function fit(){ c.width=box.clientWidth; c.height=box.clientHeight; }
addEventListener('resize',fit);

function setVol(){ if(master) master.gain.value=0.3+vol*1.7; $('vol').textContent='🔊 الصوت: '+Math.round(vol*100)+'%'; }
function audioInit(){
  if(!actx){
    actx=new (window.AudioContext||window.webkitAudioContext)();
    const comp=actx.createDynamicsCompressor(); master=actx.createGain();
    master.connect(comp); comp.connect(actx.destination); setVol();
  }
  actx.resume();
}
function strike(){
  const t=actx.currentTime;
  [[880,1],[2430,.6],[4750,.35]].forEach(([f,a])=>{
    const o=actx.createOscillator(), g=actx.createGain();
    o.type='triangle'; o.frequency.value=f;
    g.gain.setValueAtTime(a*0.5,t); g.gain.exponentialRampToValueAtTime(0.001,t+0.35);
    o.connect(g).connect(master); o.start(t); o.stop(t+0.4);
  });
}
function bell(on){
  if(on && !bellTm){ strike(); bellTm=setInterval(strike,80); }
  else if(!on && bellTm){ clearInterval(bellTm); bellTm=null; }
}
async function torch(on){
  if(on===torchOn || !stream || torchFail) return;
  try{
    await stream.getVideoTracks()[0].applyConstraints({advanced:[{torch:on}]});
    torchOn=on; $('tor').textContent='🔦 الكشاف: '+(on?'شغّال':'مطفأ');
  }catch(e){ torchFail=true; $('tor').textContent='🔦 الكشاف غير مدعوم (لازم كاميرا خلفية)'; }
}

async function start(){
  if(starting) return; starting=true;
  audioInit();
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; v.srcObject=null; await sleep(500); }
  torchOn=false; torchFail=false; $('tor').textContent='🔦 الكشاف: مطفأ';
  $('st').textContent='جاري فتح الكاميرا...';
  const tries=[{facingMode:{ideal:mode},width:{ideal:480},height:{ideal:640}},{facingMode:{ideal:mode}},true];
  let err, s=null;
  for(const vd of tries){
    try{ s=await navigator.mediaDevices.getUserMedia({audio:false,video:vd}); break; }
    catch(e){ err=e; await sleep(400); }
  }
  if(!s){ $('st').textContent='خطأ بالكاميرا: '+(err&&err.message)+' — سكّر التبويبات والتطبيقات الأخرى'; starting=false; return; }
  stream=s; v.srcObject=s;
  try{ await v.play(); }catch(e){}
  box.style.transform=(mode==='user')?'scaleX(-1)':'none';
  fit();
  cap.width=256; cap.height=Math.round(256*(v.videoHeight||4)/(v.videoWidth||3));
  target=[]; cur=[]; $('st').textContent='شغّال ✅'; starting=false;
}

function handle(d){
  if(d.length){ target=d.map(h=>h.p); miss=0; } else if(++miss>=4) target=[];
  const gs=d.map(h=>h.g);
  const g=gs.includes('fist')?'fist':gs.includes('two')?'two':gs.includes('one')?'one':d.length?'open':'none';
  if(g===last) n++; else { last=g; n=1; }
  if(n>=3) stable=g;
  bell(stable==='fist');
  if(stable==='fist') torch(false);
  if(stable==='two') torch(true);
  if(stable==='one'){
    vol=Math.min(1,vol+0.03); setVol();
    if(performance.now()-lastTick>450){ lastTick=performance.now(); strike(); }
  }
  box.classList.toggle('alarm',stable==='fist');
  $('st').textContent=TXT[stable];
  $('info').textContent='الأيدي المكتشفة: '+d.length;
}

async function loop(){
  while(true){
    if(stream && v.readyState>=2 && v.videoWidth){
      cx.drawImage(v,0,0,cap.width,cap.height);
      const b=await new Promise(r=>cap.toBlob(r,'image/jpeg',0.6));
      try{ const r=await fetch('/d',{method:'POST',body:b}); handle(await r.json()); }catch(e){}
    }
    await sleep(20);
  }
}

function draw(){
  x.clearRect(0,0,c.width,c.height);
  if(cur.length!==target.length) cur=target.map(h=>h.map(p=>[p[0],p[1]]));
  const vw=v.videoWidth||1, vh=v.videoHeight||1, W=c.width, H=c.height;
  const s=Math.max(W/vw,H/vh), ox=(W-vw*s)/2, oy=(H-vh*s)/2;
  const X=p=>p[0]*vw*s+ox, Y=p=>p[1]*vh*s+oy;
  x.strokeStyle=(stable==='fist')?'#ff2020':'#0a9d45'; x.lineWidth=4; x.lineCap='round';
  cur.forEach((h,i)=>{
    h.forEach((p,j)=>{ p[0]+=(target[i][j][0]-p[0])*0.35; p[1]+=(target[i][j][1]-p[1])*0.35; });
    x.beginPath();
    L.forEach(([a,b])=>{ x.moveTo(X(h[a]),Y(h[a])); x.lineTo(X(h[b]),Y(h[b])); });
    x.stroke(); x.fillStyle='#fff';
    h.forEach(p=>{ x.beginPath(); x.arc(X(p),Y(p),5,0,7); x.fill(); });
  });
  requestAnimationFrame(draw);
}

$('go').onclick=start;
$('cam').onclick=()=>{
  mode=(mode==='environment')?'user':'environment';
  $('cam').textContent='الكاميرا: '+(mode==='environment'?'خلفية':'أمامية')+' 🔄';
  if(stream) start();
};
setVol(); fit(); loop(); draw();
</script></body></html>"""

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

def gesture(lm):
    d = lambda a, b: np.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)
    up = [d(t, 0) > d(p, 0) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
    if not any(up): return "fist"
    if up == [True, False, False, False]: return "one"
    if up == [True, True, False, False]: return "two"
    return "open"

@app.post("/d")
def detect():
    img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
    out = []
    if img is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        for h in res.hand_landmarks:
            out.append({"p": [[round(p.x, 4), round(p.y, 4)] for p in h], "g": gesture(h)})
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, threaded=False)
