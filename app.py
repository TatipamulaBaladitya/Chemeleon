from flask import Flask, request, render_template_string, send_from_directory, url_for, session, jsonify
import cv2
import numpy as np
import os
from itertools import product
import requests
from werkzeug.utils import secure_filename
from collections import defaultdict

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this_in_prod"
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Upload route
@app.route('/uploads/<filename>', endpoint='uploaded_file')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ──────────────────────────────
# Sanzo Wada loading
# ──────────────────────────────
SANZO_COLORS = []
SANZO_PALETTES = {}

try:
    url = "https://raw.githubusercontent.com/mattdesl/dictionary-of-colour-combinations/master/colors.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    SANZO_COLORS = response.json()
except Exception as e:
    print(f"Sanzo load failed: {e}")

if SANZO_COLORS:
    for pid in range(1, 349):
        palette = [c for c in SANZO_COLORS if pid in c.get('combinations', [])]
        if palette:
            SANZO_PALETTES[pid] = [{'name': c['name'], 'hex': c['hex']} for c in palette]

# ──────────────────────────────
# COLOR LOGIC FUNCTIONS
# ──────────────────────────────
def get_skin_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 30, 60), (20, 170, 255))

def detect_skin_tone(path):
    img = cv2.imread(path)
    if img is None:
        return "Unknown"
    img = cv2.resize(img, (200, 200))
    mask = get_skin_mask(img)
    skin = img[mask > 0]
    if len(skin) == 0:
        return "Unknown"
    brightness = np.mean(skin)
    if brightness > 180: return "Fair"
    if brightness > 140: return "Medium"
    if brightness > 100: return "Olive"
    return "Dark"

COLOR_DB = {
    "Black": (20,20,20), "White": (240,240,240),
    "Navy": (20,40,80), "Blue": (60,100,180),
    "Green": (50,120,60), "Olive": (85,107,47),
    "Beige": (245,245,200), "Brown": (101,67,33),
    "Mustard": (205,173,0), "Red": (160,40,40)
}

def dominant_color(path):
    img = cv2.imread(path)
    if img is None:
        return "Uncertain"
    img = cv2.resize(img, (150,150))
    avg = np.mean(img.reshape(-1,3), axis=0)
    best, dist = None, 1e9
    for name, ref in COLOR_DB.items():
        d = np.linalg.norm(avg - np.array(ref))
        if d < dist:
            best, dist = name, d
    return best if dist < 70 else "Uncertain"

RULES = {
    "Fair": ["Navy", "Blue"],
    "Medium": ["Blue", "Green"],
    "Olive": ["Beige", "Mustard"],
    "Dark": ["White", "Red"]
}

FACE_SHAPE_RULES = {
    "Oval": "Most styles work well",
    "Square": "Open collars and layered outfits suit you",
    "Diamond": "Structured shoulders look best",
    "Rectangular": "Contrast tops balance face length"
}

def find_best_sanzo_palette(top_color, bottom_color):
    if not SANZO_PALETTES:
        return ["#cccccc", "#aaaaaa"], "Sanzo data unavailable"
    candidates = []
    for pid, palette in SANZO_PALETTES.items():
        names = [c['name'].lower() for c in palette]
        if any(top_color.lower() in n or bottom_color.lower() in n for n in names):
            hexes = [c['hex'] for c in palette]
            candidates.append((len(palette), hexes, pid))
    if candidates:
        candidates.sort(reverse=True)
        _, hexes, pid = candidates[0]
        return hexes, f"Sanzo Wada Palette #{pid}"
    return ["#e0d4b8", "#a68a64"], "Natural tones"

# 🔧 FIXED: preserve all clothes (NO LOGIC CHANGE)
def generate_all_pairings(skin_tone, face_shape, tops_dict, bottoms_dict):
    preferred_tops = [c for c in tops_dict if c in RULES.get(skin_tone, [])]
    if not preferred_tops:
        preferred_tops = list(tops_dict.keys())

    pairings = []
    for top in preferred_tops:
        for bottom in bottoms_dict:
            for top_img in tops_dict[top]:
                for bottom_img in bottoms_dict[bottom]:
                    palette_colors, sanzo_note = find_best_sanzo_palette(top, bottom)
                    pairings.append({
                        "top_color": top,
                        "top_img": url_for('uploaded_file', filename=os.path.basename(top_img)),
                        "bottom_color": bottom,
                        "bottom_img": url_for('uploaded_file', filename=os.path.basename(bottom_img)),
                        "reason": f"{top} suits {skin_tone} skin • {FACE_SHAPE_RULES.get(face_shape)} • {sanzo_note}",
                        "palette_colors": palette_colors
                    })
    return pairings

# ──────────────────────────────
# API Endpoints
# ──────────────────────────────
@app.route('/api/save_face_shape', methods=['POST'])
def save_face_shape():
    session['face_shape'] = request.json.get('face_shape')
    return jsonify({"status": "ok"})

@app.route('/api/upload_face', methods=['POST'])
def upload_face():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file"}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    skin_tone = detect_skin_tone(path)
    session['skin_tone'] = skin_tone
    return jsonify({
        "success": True,
        "skin_tone": skin_tone,
        "preview_url": url_for('uploaded_file', filename=filename)
    })

@app.route('/api/upload_clothes', methods=['POST'])
def upload_clothes():
    files = request.files.getlist('files')
    item_type = request.form.get('type')   # 'tops' or 'bottoms'

    uploaded = session.get(item_type, [])

    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            uploaded.append({"path": path, "color": dominant_color(path)})

    session[item_type] = uploaded
    return jsonify({"status": "ok", "count": len(files)})


@app.route('/api/generate', methods=['GET'])
def generate():
    skin_tone = session.get('skin_tone', 'Unknown')
    face_shape = session.get('face_shape', 'Oval')

    tops_dict = defaultdict(list)
    bottoms_dict = defaultdict(list)

    for i in session.get('tops', []):
        tops_dict[i['color']].append(i['path'])
    for i in session.get('bottoms', []):
        bottoms_dict[i['color']].append(i['path'])

    pairings = generate_all_pairings(skin_tone, face_shape, tops_dict, bottoms_dict)
    return jsonify({"pairings": pairings, "skin_tone": skin_tone})

# ──────────────────────────────
# FULL ORIGINAL HTML (UNCHANGED)
# ──────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Chameleon</title>
  <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap" rel="stylesheet">
  

  <style>
    :root { --bg: #f5f5f7; --card: #fff; --text: #1d1d1f; --accent: #007aff; --gray: #8e8e93; }
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family: 'SF Pro Display', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    .screen { position: absolute; inset: 0; padding: 2rem; display: flex; flex-direction: column; justify-content: center; align-items: center; transition: opacity 0.5s ease, transform 0.5s ease; }
    .hidden { opacity: 0; pointer-events: none; transform: translateX(30px); }
    h1 { font-size: 2.8rem; font-weight: 600; margin-bottom: 1rem; }
    p { font-size: 1.25rem; color: var(--gray); text-align: center; max-width: 480px; margin-bottom: 2.5rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1.5rem; width: 100%; max-width: 640px; }
    .card { background: var(--card); border-radius: 20px; overflow: hidden; box-shadow: 0 4px 25px rgba(0,0,0,0.08); cursor: pointer; transition: all 0.3s; aspect-ratio: 1; position: relative; }
    .card.selected { border: 4px solid var(--accent); transform: scale(1.06); box-shadow: 0 12px 40px rgba(0,122,255,0.25); }
    .card img { width: 100%; height: 100%; object-fit: cover; }
    .label { position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.6); color: white; padding: 6px 14px; border-radius: 20px; font-size: 0.9rem; }
    button { background: var(--accent); color: white; border: none; padding: 1rem 2.5rem; font-size: 1.15rem; border-radius: 999px; cursor: pointer; margin-top: 2rem; transition: all 0.3s; }
    button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,122,255,0.35); }
    button:disabled { background: #c7c7cc; cursor: not-allowed; }
    .back { position: absolute; top: 1.5rem; left: 1.5rem; background: none; border: none; color: var(--accent); font-size: 1.3rem; cursor: pointer; }
    .upload-box { background: var(--card); border-radius: 20px; padding: 3rem 2rem; text-align: center; box-shadow: 0 4px 25px rgba(0,0,0,0.08); width: 100%; max-width: 420px; }
    .preview { display: flex; flex-wrap: wrap; gap: 0.8rem; margin-top: 1.5rem; justify-content: center; }
    .chip { background: rgba(0,122,255,0.1); color: var(--accent); padding: 0.5rem 1rem; border-radius: 999px; font-size: 0.9rem; }

    /*Appending*/
    /* ===== Left → Right Screen Flow (ADD ONLY) ===== */

.screen {
  transition: transform 0.45s ease, opacity 0.45s ease;
}

.hidden {
  transform: translateX(120%);
  opacity: 0;
  pointer-events: none;
}

.screen:not(.hidden) {
  transform: translateX(0%);
}

/* ===== Progress Bar (ADD ONLY) ===== */

.progress {
  position: fixed;
  top: 0;
  left: 0;
  height: 4px;
  width: 100%;
  background: #e5e7eb;
  z-index: 1000;
}

.progress .bar {
  height: 100%;
  width: 0%;
  background: var(--accent, #4f46e5);
  transition: width 0.4s ease;
}


/*CAMERA*/
/* Camera button beside Next */
.action-row {
  display: flex;
  gap: 10px;
  margin-top: 16px;
}

.camera-btn {
  flex: 0 0 52px;
  border-radius: 12px;
  border: none;
  background: #e5e7eb;
  font-size: 20px;
  cursor: pointer;
}


  </style>
</head>
<body>
<input
  type="file"
  id="cameraInput"
  accept="image/*"
  capture="environment"
  style="display:none">

<div class="progress"><div class="bar" id="progressBar"></div></div>
<!-- Step 1: Welcome -->
<div id="welcome" class="screen">
  <h1>Chameleon 🦎</h1>
  <p>Discover perfect outfits with AI and timeless Sanzo Wada color harmony</p>
  <button onclick="nextStep('faceShape')">Get Started</button>
</div>

<!-- Step 2: Face Shape -->
<div id="faceShape" class="screen hidden">
  <button class="back" onclick="prevStep()">←</button>
  <h1>Your Face Shape</h1>
  <p>Tap the one that looks like you</p>
  <div class="grid">
    <div class="card" onclick="select('Oval', this)"><img src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300" alt="Oval"><div class="label">Oval</div></div>
    <div class="card" onclick="select('Square', this)"><img src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=300" alt="Square"><div class="label">Square</div></div>
    <div class="card" onclick="select('Diamond', this)"><img src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=300" alt="Diamond"><div class="label">Diamond</div></div>
    <div class="card" onclick="select('Rectangular', this)"><img src="https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=300" alt="Rectangular"><div class="label">Rectangular</div></div>
  </div>
  <button id="next1" disabled onclick="nextStep('facePhoto')">Next</button>
</div>

<!-- Step 3: Face Photo -->
<div id="facePhoto" class="screen hidden">
  <button class="back" onclick="prevStep()">←</button>
  <h1>Upload Your Face</h1>
  <p>We'll detect your skin tone</p>
  <div class="upload-box">
    <label for="faceInput" class="upload-btn">Choose Photo</label>
    <input type="file" id="faceInput" accept="image/*" onchange="uploadFile(this)">
    <div id="facePreview" class="preview"></div>
  </div>
  <button id="next2" disabled onclick="nextStep('tops')">Next</button>
  
</div>

<!-- Step 4: Tops -->
<div id="tops" class="screen hidden">
  <button class="back" onclick="prevStep()">←</button>
  <h1>Upload Your Tops</h1>
  <p>Multiple allowed</p>
  <div class="upload-box">
    <label for="topsInput" class="upload-btn">Choose Tops</label>
    <input type="file" id="topsInput" multiple accept="image/*" onchange="uploadMultiple(this, 'tops')">
    <div id="topsPreview" class="preview"></div>
  </div>
  <button id="next3" onclick="nextStep('bottoms')">Next</button>
  <div class="action-row">
  <button onclick="nextStep('bottoms')">Continue</button>

  <button class="camera-btn" onclick="openCamera('top')">
    📷
  </button>
</div>

</div>

<!-- Step 5: Bottoms -->
<div id="bottoms" class="screen hidden">
  <button class="back" onclick="prevStep()">←</button>
  <h1>Upload Your Bottoms</h1>
  <p>Multiple allowed</p>
  <div class="upload-box">
    <label for="bottomsInput" class="upload-btn">Choose Bottoms</label>
    <input type="file" id="bottomsInput" multiple accept="image/*" onchange="uploadMultiple(this, 'bottoms')">
    <div id="bottomsPreview" class="preview"></div>
  </div>
  <button id="next4" onclick="generateResults()">See Suggestions</button>

  <div class="action-row">
  <button onclick="showResults()">Show My Outfits</button>

  <button class="camera-btn" onclick="openCamera('bottom')">
    📷
  </button>
</div>

</div>

<!-- Step 6: Color Recommendations -->
<div id="recommendations" class="screen hidden">
  <button class="back" onclick="prevStep()">← Back</button>
  <h1>Sanzo Wada Recommendations</h1>
  <p>Colors that suit your skin tone and face shape</p>

  <!-- Add headings for clarity here -->
  <h3 style="margin-top:1rem;">Suggested Tops</h3>
  <div id="recommendationTops" style="margin-top:0.5rem; display:flex; flex-wrap:wrap; gap:12px;"></div>

  <h3 style="margin-top:1rem;">Suggested Bottoms</h3>
  <div id="recommendationBottoms" style="margin-top:0.5rem; display:flex; flex-wrap:wrap; gap:12px;"></div>
</div>



<!-- Results -->
<div id="results" class="screen hidden">
  <button class="back" onclick="prevStep()">← Back</button>
  <h1>Your Outfits</h1>
  <div id="resultsContent"></div>
</div>


<script>
let stepHistory = ['welcome'];
let selections = {};

function nextStep(id) {
  document.getElementById(stepHistory[stepHistory.length-1]).classList.add('hidden');
  document.getElementById(id).classList.remove('hidden');
  stepHistory.push(id);
}

function prevStep() {
  if (stepHistory.length > 1) {
    document.getElementById(stepHistory.pop()).classList.add('hidden');
    document.getElementById(stepHistory[stepHistory.length-1]).classList.remove('hidden');
  }
}

function select(shape, el) {
  document.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  selections.faceShape = shape;
  document.getElementById('next1').disabled = false;
  fetch('/api/save_face_shape', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({face_shape: shape})
  });
}

async function uploadFile(input) {
  const file = input.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload_face', {method: 'POST', body: formData});
    const data = await res.json();

    if (data.success) {
      document.getElementById('facePreview').innerHTML = 
        `<img src="${data.preview_url}" style="max-width:200px;border-radius:12px;">`;
      document.getElementById('next2').disabled = false;
      alert(`Skin tone detected: ${data.skin_tone}`);
    } else {
      alert('Error: ' + (data.error || 'Try again'));
    }
  } catch (err) {
    console.error('Fetch error:', err);
    alert('Error: ' + err.message);
  }
}

async function uploadMultiple(input, type) {
  const files = input.files;
  if (!files.length) return;

  const formData = new FormData();
  for (let file of files) formData.append('files', file);
  formData.append('type', type);

  try {
    const res = await fetch('/api/upload_clothes', {method: 'POST', body: formData});
    const data = await res.json();

    if (data.status === "ok") {
      const preview = document.getElementById(type + 'Preview');
      preview.innerHTML = '';
      for (let file of files) {
        preview.innerHTML += `<div class="chip">${file.name}</div>`;
      }
      alert(`Uploaded ${data.count} ${type}(s)!`);
    } else {
      alert('Error: ' + (data.error || 'Try again'));
    }
  } catch (err) {
    console.error('Fetch error:', err);
    alert('Error: ' + err.message);
  }
}

async function generateResults() {
  const res = await fetch('/api/generate');
  const data = await res.json();
  const content = document.getElementById('resultsContent');
  content.innerHTML = '';
  if (data.pairings.length === 0) {
    content.innerHTML = '<p style="color:#8e8e93;">Add tops & bottoms to see suggestions ✨</p>';
  } else {
    data.pairings.forEach(p => {
      let swatches = p.palette_colors.map(hex => `<div class="swatch" style="background:${hex}"></div>`).join('');
      content.innerHTML += `
        <div style="margin:2rem 0; background:white; border-radius:16px; padding:1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.08);">
          <div style="display:flex; gap:1rem; justify-content:center; margin-bottom:1rem;">
            <img src="${p.top_img}" width="140" style="border-radius:12px;">
            <img src="${p.bottom_img}" width="140" style="border-radius:12px;">
          </div>
          <p style="font-size:1.1rem;">${p.reason}</p>
          <div style="display:flex; gap:10px; justify-content:center; margin-top:1rem;">${swatches}</div>
        </div>`;
    });

  }
  nextStep('results');



data.pairings.forEach(p => {
  let swatches = p.palette_colors.map(hex => `
    <div style="
      width:36px;
      height:36px;
      border-radius:6px;
      background:${hex};
      margin-right:6px;
      flex:0 0 auto;
    " title="${hex}"></div>
  `).join('');

  content.innerHTML += `
    <div style="margin:2rem 0; background:white; border-radius:16px; padding:1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.08);">
      <div style="display:flex; gap:1rem; justify-content:center; margin-bottom:1rem;">
        <img src="${p.top_img}" width="140" style="border-radius:12px;">
        <img src="${p.bottom_img}" width="140" style="border-radius:12px;">
      </div>
      <p style="font-size:1.1rem;">${p.reason}</p>
      <div style="display:flex; overflow-x:auto; gap:6px; padding-top:0.5rem;">${swatches}</div>
    </div>`;
});


}


/* ===== Progress Tracking (ADD ONLY) ===== */

const __steps = ['welcome','tops','bottoms','results'];

const __oldNextStep = window.nextStep;
window.nextStep = function(id) {
  __oldNextStep(id);
  updateProgress();
};

function updateProgress() {
  const current = stepHistory[stepHistory.length - 1];
  const index = __steps.indexOf(current);
  if (index >= 0) {
    document.getElementById('progressBar').style.width =
      ((index + 1) / __steps.length) * 100 + '%';
  }
}

function openCamera(type) {
  const cam = document.getElementById('cameraInput');

  cam.onchange = function () {
    uploadFiles(cam, type);
    cam.value = ''; // reset
  };

  cam.click();
}


function generateRecommendations() {
  const skinTone = sessionStorage.getItem('skin_tone') || 'Unknown';
  const faceShape = sessionStorage.getItem('face_shape') || 'Oval';

  // Collect colors from uploaded clothes
  const tops = Array.from(document.querySelectorAll('#topsPreview .chip')).map(c => c.innerText);
  const bottoms = Array.from(document.querySelectorAll('#bottomsPreview .chip')).map(c => c.innerText);

  const recommendationTops = document.getElementById('recommendationTops');
  const recommendationBottoms = document.getElementById('recommendationBottoms');

  recommendationTops.innerHTML = '';
  recommendationBottoms.innerHTML = '';

  // Loop through SANZO_PALETTES to find feasible colors
  for (const pid in SANZO_PALETTES) {
    const palette = SANZO_PALETTES[pid];
    palette.forEach(c => {
      const colorDiv = `<div style="width:40px;height:40px;border-radius:8px;background:${c.hex};display:flex;align-items:center;justify-content:center;color:white;font-size:10px;">${c.name}</div>`;
      // Add to tops if top matches skin tone rules
      if (!tops.includes(c.name)) recommendationTops.innerHTML += colorDiv;
      // Add to bottoms if bottom matches skin tone rules
      if (!bottoms.includes(c.name)) recommendationBottoms.innerHTML += colorDiv;
    });
  }

  nextStep('recommendations');
}

</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

if __name__ == "__main__":
    app.run(debug=True)
