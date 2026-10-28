"""
The blind A/B of the golden frames (docs/FORMS.md §3, the brief's fifth test): which of two frames is
closer to the TotK reference. `build` writes one self-contained page, `tally` reads the players'
answers. PIL only.

  python3 -I bench/forms/abkit.py build <before dir> <after dir> [--ref=<dir>] --out=<page.html>
  python3 -I bench/forms/abkit.py tally <answers.json> …

The frames are the stills runner's game-med_12_fNN.png (bench/forms/sheet.py has the layout). --ref
names a local folder of TotK captures, p1.jpg … p5.jpg, shown above each pair; the repository holds
none, and without them a player judges against their own memory of TotK. Each player gets the 15
views in a fresh random order, each with the two builds on random sides, and saves a JSON file of
answers at the end. `tally` gives the after build's share of the comparisons, per place and over all,
with a one-sided binomial p against a coin; the brief asks for at least five players and 70 %.
"""
import base64, io, json, math, os, sys
from PIL import Image

PLACES = ['p1 jungle karst', 'p2 mountain flank', 'p3 banks and ponds', 'p4 sea cliff', 'p5 karst towers']
DIST = ['near', 'mid', 'far']


def arg(k, d):
    return next((a[len(k) + 3:] for a in sys.argv if a.startswith(f'--{k}=')), d)


def jpeg(im, w=None):
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    im.convert('RGB').save(b, 'JPEG', quality=86)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


def build(before, after):
    rdir = arg('ref', '')
    views = []
    for pi, place in enumerate(PLACES):
        r = next((os.path.join(rdir, f'p{pi + 1}.{e}') for e in ('jpg', 'jpeg', 'png', 'webp')
                  if rdir and os.path.exists(os.path.join(rdir, f'p{pi + 1}.{e}'))), None)
        for di, dist in enumerate(DIST):
            k = 3 * pi + di
            fb, fa = (os.path.join(d, f'game-med_12_f{k:02d}.png') for d in (before, after))
            if not (os.path.exists(fb) and os.path.exists(fa)):
                print(f'skip f{k:02d}: not rendered in both builds')
                continue
            views.append({'id': f'f{k:02d}', 'place': place, 'dist': dist,
                          'b': jpeg(Image.open(fb)), 'a': jpeg(Image.open(fa)),
                          'ref': jpeg(Image.open(r), 960) if r else None})
    out = arg('out', '')
    with open(out, 'w') as f:
        f.write(PAGE.replace('/*VIEWS*/[]', json.dumps(views)))
    print(f'{out}: {len(views)} views, {"with" if rdir else "without"} references, {os.path.getsize(out) / 1e6:.1f} MB')


def tally(files):
    rows = []
    players = set()
    for p in files:
        d = json.load(open(p))
        players.add(d.get('player') or p)
        rows += [(a['id'], a['place'], a['pick'] == a['after']) for a in d['answers']]

    def line(name, wins):
        n, k = len(wins), sum(wins)
        # one-sided binomial: the chance a coin wins k or more of n
        pv = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0
        return f'{name:<20} {k:>3}/{n:<3} after {100 * k / max(1, n):5.1f} %  p {pv:.4f}'

    print(f'{len(players)} players, {len(rows)} comparisons')
    for place in PLACES:
        print(line(place, [w for _, pl, w in rows if pl == place]))
    allw = [w for *_, w in rows]
    print(line('all', allw))
    ok = len(players) >= 5 and allw and sum(allw) / len(allw) >= 0.7
    print('brief: ≥ 5 players and ≥ 70 % for the after build →', 'met' if ok else 'not met')


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Terrain A/B</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mute:#6b6a66;--line:#d9d7d1;--accent:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe7;--mute:#9b9a95;--line:#3a3936;--accent:#8db4dc}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1320px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}p{margin:0 0 12px;color:var(--mute)}
.ref{text-align:center;margin:8px 0 12px}.ref img{max-width:min(960px,100%);border-radius:6px}
.ref .none{padding:24px;border:1px dashed var(--line);border-radius:6px;color:var(--mute)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:720px){.pair{grid-template-columns:1fr}}
button.pick{all:unset;cursor:pointer;display:block;border:2px solid var(--line);border-radius:8px;overflow:hidden}
button.pick:hover,button.pick:focus-visible{border-color:var(--accent)}
button.pick img{display:block;width:100%;height:auto}
button.pick span{display:block;padding:6px 10px;font-weight:600}
.bar{display:flex;gap:12px;align-items:center;margin:12px 0}
input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit}
button.go{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:var(--bg);cursor:pointer}
</style></head><body><main id="app"></main>
<script>
const VIEWS = /*VIEWS*/[];
const app = document.getElementById('app');
const rnd = (n) => crypto.getRandomValues(new Uint32Array(1))[0] % n;
let player = '', order = [], at = 0, answers = [];
function start() {
  app.innerHTML = `<h1>Which frame is closer to Tears of the Kingdom?</h1>
    <p>${VIEWS.length} views. Each shows two frames of the same place from the same camera; pick the one whose cliffs,
    towers, walls and slopes read more like TotK's${VIEWS.some((v) => v.ref) ? ', using the capture above them as the reference' : ''}.
    Judge the landforms, not the sky or the trees.</p>
    <div class="bar"><input id="who" placeholder="Your name or initials"><button class="go" id="go">Start</button></div>`;
  document.getElementById('go').onclick = () => {
    player = document.getElementById('who').value.trim() || 'anonymous';
    order = VIEWS.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) { const j = rnd(i + 1); [order[i], order[j]] = [order[j], order[i]]; }
    at = 0; answers = []; show();
  };
}
function show() {
  if (at >= order.length) return done();
  const v = VIEWS[order[at]], afterSide = rnd(2) ? 'A' : 'B';
  const src = { A: afterSide === 'A' ? v.a : v.b, B: afterSide === 'B' ? v.a : v.b };
  app.innerHTML = `<h1>${at + 1} of ${order.length}</h1>
    <div class="ref">${v.ref ? `<img alt="TotK reference" src="${v.ref}">` : '<div class="none">No reference capture: judge against TotK as you remember it.</div>'}</div>
    <div class="pair">${['A', 'B'].map((s) => `<button class="pick" data-s="${s}"><img alt="Frame ${s}" src="${src[s]}"><span>${s} is closer</span></button>`).join('')}</div>`;
  app.querySelectorAll('button.pick').forEach((b) => b.onclick = () => {
    answers.push({ id: v.id, place: v.place, dist: v.dist, pick: b.dataset.s, after: afterSide });
    at++; show();
  });
}
function done() {
  const data = JSON.stringify({ player, when: new Date().toISOString(), answers }, null, 1);
  const url = URL.createObjectURL(new Blob([data], { type: 'application/json' }));
  const name = `ab-${player.replace(/[^\w-]+/g, '_')}.json`;
  app.innerHTML = `<h1>Thank you</h1><p>Save your answers and send the file back.</p>
    <div class="bar"><a class="go" download="${name}" href="${url}" style="text-decoration:none;padding:6px 14px;border-radius:6px;background:var(--accent);color:var(--bg)">Save ${name}</a></div>`;
}
start();
</script></body></html>
"""

if __name__ == '__main__':
    cmd, pos = (sys.argv[1] if len(sys.argv) > 1 else ''), [a for a in sys.argv[2:] if not a.startswith('--')]
    if cmd == 'build' and len(pos) == 2 and arg('out', ''):
        build(*pos)
    elif cmd == 'tally' and pos:
        tally(pos)
    else:
        sys.exit(__doc__)
