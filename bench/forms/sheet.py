"""
A contact sheet of the golden frames (bench/forms/frames.ts, `webgpu.ts stills --config=forms`): one
row per place, its near, mid and far views each as a before | after pair, and a last column for the
matching TotK reference. The repository holds no TotK captures, so --ref names a local folder of
them (p1.jpg … p5.jpg, or .png), and without it the column says so. PIL only.

  python3 -I bench/forms/sheet.py <before dir> <after dir> [--ref=<dir>] [--scale=0.5] --out=<file.jpg>

The frames are the stills runner's game-med_12_fNN.png, three per place in place order: f00–f02 are
place 1 near, mid and far, f03–f05 place 2, and so on.
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

PLACES = ['p1 jungle karst', 'p2 mountain flank', 'p3 banks and ponds', 'p4 sea cliff', 'p5 karst towers']
DIST = ['near', 'mid', 'far']


def arg(k, d):
    return next((a[len(k) + 3:] for a in sys.argv if a.startswith(f'--{k}=')), d)


def frame(d, k):
    p = os.path.join(d, f'game-med_12_f{k:02d}.png')
    return Image.open(p).convert('RGB') if os.path.exists(p) else None


def ref(d, i):
    for ext in ('jpg', 'jpeg', 'png', 'webp'):
        p = os.path.join(d, f'p{i + 1}.{ext}')
        if d and os.path.exists(p):
            return Image.open(p).convert('RGB')
    return None


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(pos) != 2 or not arg('out', ''):
        sys.exit(__doc__)
    before, after = pos
    scale, rdir = float(arg('scale', '0.5')), arg('ref', '')
    probe = frame(before, 0)
    probe = probe if probe is not None else frame(after, 0)
    w, h = round(probe.width * scale), round(probe.height * scale)
    gap, head, side = 4, 22, 150
    cols = 2 * len(DIST) + 1
    W = side + cols * w + (cols + len(DIST)) * gap
    H = head * 2 + len(PLACES) * (h + gap)
    sheet = Image.new('RGB', (W, H), (24, 24, 24))
    dr = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(14)

    def x0(c):
        # (a wider gap after each before | after pair)
        return side + c * (w + gap) + (c // 2) * gap

    for di, dn in enumerate(DIST):
        dr.text((x0(2 * di), 4), dn, fill=(235, 235, 235), font=font)
        dr.text((x0(2 * di), head + 2), 'before', fill=(170, 170, 170), font=font)
        dr.text((x0(2 * di + 1), head + 2), 'after', fill=(170, 170, 170), font=font)
    dr.text((x0(cols - 1), head + 2), 'TotK reference', fill=(170, 170, 170), font=font)
    for pi, name in enumerate(PLACES):
        y = head * 2 + pi * (h + gap)
        dr.text((8, y + h // 2 - 8), name, fill=(235, 235, 235), font=font)
        for di in range(len(DIST)):
            for si, d in enumerate((before, after)):
                im = frame(d, 3 * pi + di)
                c = 2 * di + si
                if im is None:
                    dr.rectangle([x0(c), y, x0(c) + w - 1, y + h - 1], outline=(90, 90, 90))
                    dr.text((x0(c) + 8, y + h // 2 - 8), 'not rendered', fill=(140, 140, 140), font=font)
                else:
                    sheet.paste(im.resize((w, h), Image.LANCZOS), (x0(c), y))
        r, c = ref(rdir, pi), cols - 1
        if r is None:
            dr.rectangle([x0(c), y, x0(c) + w - 1, y + h - 1], outline=(90, 90, 90))
            dr.text((x0(c) + 8, y + h // 2 - 16), 'no capture in the repo:', fill=(140, 140, 140), font=font)
            dr.text((x0(c) + 8, y + h // 2 + 2), f'--ref=<dir>/p{pi + 1}.jpg', fill=(140, 140, 140), font=font)
        else:
            # (fit inside the cell, centred, keeping the capture's own aspect)
            k = min(w / r.width, h / r.height)
            r = r.resize((round(r.width * k), round(r.height * k)), Image.LANCZOS)
            sheet.paste(r, (x0(c) + (w - r.width) // 2, y + (h - r.height) // 2))
    out = arg('out', '')
    sheet.save(out, quality=88) if out.lower().endswith(('.jpg', '.jpeg')) else sheet.save(out)
    print(f'{out}: {W}×{H}')


if __name__ == '__main__':
    main()
