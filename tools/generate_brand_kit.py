# tools/generate_brand_kit.py
# EdgeLine Brand Kit generator — logos, icons, splash, animation, zip
# Uses PIL + moviepy. Outputs artifacts/EdgeLine_Press_Kit.zip

import os, io, math, zipfile, shutil, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# ---- Brand constants (locked to your decisions) ----
BG_BLACK = (10,10,10)            # #0A0A0A rich matte
WHITE = (255,255,255)
GOLD_WARM = (217,164,65)         # #D9A441 warm rich yellow gold
GOLD_DEEP = (161,118,43)         # companion tone for gradient
GREEN_EMERALD = (0,255,136)      # #00FF88 tagline (bright emerald, clean)
GREEN_EMERALD_DARK = (0,200,110)

WORDMARK = "EdgeLine"
TAGLINE  = "Predict. Play. Profit."

# Paths
ROOT = Path(".").resolve()
OUT  = ROOT / "artifacts"
(OUT / "logos").mkdir(parents=True, exist_ok=True)
(OUT / "icons").mkdir(parents=True, exist_ok=True)
(OUT / "splash").mkdir(parents=True, exist_ok=True)
(OUT / "animation").mkdir(parents=True, exist_ok=True)

# Try to load a nice sans font; fall back to PIL default
def load_font(size:int, bold:bool=False, italic:bool=False):
    # Try common fonts on CI
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                pass
    # Fallback
    from PIL import ImageFont as _F
    return _F.load_default()

# --------- Helpers ----------
def lerp(a,b,t): return a + (b - a)*t

def gradient_image(w,h, start_rgb, end_rgb, angle_deg=45):
    # Create diagonal linear gradient
    img = Image.new("RGB", (w,h))
    arr = np.zeros((h,w,3), dtype=np.uint8)
    # normalized coordinates rotated by angle
    th = math.radians(angle_deg)
    cos, sin = math.cos(th), math.sin(th)
    cx, cy = w/2.0, h/2.0
    for y in range(h):
        dy = (y - cy)
        for x in range(w):
            dx = (x - cx)
            t = (dx*cos + dy*sin) / (max(w,h)/2.0)
            t = (t + 1)/2
            t = max(0.0, min(1.0, t))
            r = int(lerp(start_rgb[0], end_rgb[0], t))
            g = int(lerp(start_rgb[1], end_rgb[1], t))
            b = int(lerp(start_rgb[2], end_rgb[2], t))
            arr[y,x] = (r,g,b)
    return Image.fromarray(arr, mode="RGB")

def draw_wordmark(canvas:Image.Image, text:str, color, y:int, weight=700, letter_spacing=0, size=140):
    draw = ImageDraw.Draw(canvas)
    font = load_font(size, bold=True)
    # crude letter spacing
    x = canvas.width//2
    w,h = draw.textsize(text, font=font)
    # center baseline
    curx = x - w//2
    draw.text((curx,y), text, font=font, fill=color)
    return (x - w//2, y, w, h)

def draw_tagline(canvas:Image.Image, text:str, color, y:int, size=56):
    draw = ImageDraw.Draw(canvas)
    font = load_font(size, bold=False)
    w,h = draw.textsize(text, font=font)
    x = (canvas.width - w)//2
    draw.text((x,y), text, font=font, fill=color)
    return (x,y,w,h)

def draw_icon_E_with_arrow(size:int, gold_light=GOLD_WARM, gold_dark=GOLD_DEEP, bg=None, rounded=True):
    # Square icon with stylized "E" and upward arrow (cut-through)
    img = Image.new("RGBA", (size,size), (0,0,0,0) if bg is None else (*bg,255))
    d = ImageDraw.Draw(img)

    # Background gradient if bg is a gradient request
    if bg == "gold_gradient":
        g = gradient_image(size, size, GOLD_DARK, GOLD_WARM, angle_deg=45)
        img.paste(g)

    # Stylized E (three bars)
    pad = int(size*0.18)
    bar_h = int(size*0.12)
    gap   = int(size*0.08)
    x0 = pad; x1 = size - pad
    y_mid = size//2

    # Bars: top/mid/bottom in gold
    for i, yy in enumerate([pad, y_mid - bar_h//2, size - pad - bar_h]):
        d.rounded_rectangle([x0, yy, x1, yy+bar_h], radius=bar_h//2 if rounded else 0, fill=GOLD_WARM)

    # Upward arrow (cut-through): draw a slightly darker path then "erase" line
    # path from bottom-left quadrant to top-right
    ax0, ay0 = int(size*0.26), int(size*0.68)
    ax1, ay1 = int(size*0.78), int(size*0.28)
    # draw stroke by compositing: first erase a corridor, then draw green arrow
    stroke_w = max(3, size//18)
    erase = Image.new("RGBA", (size,size), (0,0,0,0))
    ed = ImageDraw.Draw(erase)
    ed.line([(ax0,ay0),(ax1,ay1)], fill=(0,0,0,0), width=stroke_w)
    # punch alpha corridor
    img = Image.alpha_composite(img, erase)

    # arrow (subtle highlight)
    d.line([(ax0,ay0),(ax1,ay1)], fill=(255,255,255,180), width=stroke_w//2)
    # arrow head
    ah = stroke_w*2
    d.polygon([(ax1,ay1),
               (ax1-ah, ay1+ah//2),
               (ax1-ah, ay1-ah//2)], fill=(255,255,255,200))

    return img

def make_full_logo(bg_mode="black", with_tagline=True, transparent=False):
    W,H = 2400, 1350  # 16:9 for splash/press, very high-res
    if transparent:
        canvas = Image.new("RGBA", (W,H), (0,0,0,0))
    else:
        if bg_mode == "black":
            canvas = Image.new("RGB", (W,H), BG_BLACK)
        elif bg_mode == "gold_gradient":
            canvas = gradient_image(W,H, GOLD_DEEP, GOLD_WARM, angle_deg=45)
        elif bg_mode == "white":
            canvas = Image.new("RGB", (W,H), (255,255,255))
        else:
            canvas = Image.new("RGB", (W,H), BG_BLACK)

    # Icon
    icon_size = 420
    icon = draw_icon_E_with_arrow(icon_size, bg=None)
    # subtle glow behind icon
    glow = Image.new("RGBA",(icon_size*2,icon_size*2),(0,0,0,0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([0,0,glow.width, glow.height], fill=(*GOLD_WARM,60))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=28))
    ix = 260; iy = (H - icon_size)//2 - 40
    if canvas.mode == "RGB":
        overlay = Image.new("RGBA", canvas.size, (0,0,0,0))
        overlay.paste(glow, (ix-icon_size//2, iy-icon_size//2), glow)
        overlay.paste(icon, (ix,iy), icon)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    else:
        canvas.paste(glow, (ix-icon_size//2, iy-icon_size//2), glow)
        canvas.paste(icon, (ix,iy), icon)

    # Wordmark
    word_y = int(H*0.36)
    draw_wordmark(canvas, WORDMARK, WHITE if bg_mode!="white" else (0,0,0), y=word_y, size=180)

    # Tagline
    if with_tagline:
        tag_y = word_y + 220
        green = GREEN_EMERALD if bg_mode!="white" else (0,160,90)
        draw_tagline(canvas, TAGLINE, green, y=tag_y, size=70)

    return canvas

def save_png(img:Image.Image, path:Path):
    img.save(str(path), format="PNG", optimize=True)

def build_all():
    # Logos
    save_png(make_full_logo("black", with_tagline=True), OUT/"logos/logo_black.png")
    save_png(make_full_logo("gold_gradient", with_tagline=True), OUT/"logos/logo_gold_gradient.png")
    save_png(make_full_logo("black", with_tagline=True, transparent=True), OUT/"logos/logo_transparent.png")
    save_png(make_full_logo("white", with_tagline=True), OUT/"logos/logo_light.png")

    # Icons
    for sz in [1024,512,256,128]:
        icon = draw_icon_E_with_arrow(sz, bg="gold_gradient")
        save_png(icon, OUT/f"icons/app_icon_{sz}.png")

    # Splash (portrait common sizes)
    splash = make_full_logo("gold_gradient", with_tagline=True)
    # resize to a few typical sizes
    for wh in [(1284,2778),(1170,2532),(1242,2688),(1080,1920)]:
        save_png(splash.resize(wh, Image.LANCZOS), OUT/f"splash/splash_{wh[0]}x{wh[1]}.png")

    # Lightweight animation (MoviePy optional)
    try:
        from moviepy.editor import ImageClip, concatenate_videoclips, CompositeVideoClip, vfx
        base = make_full_logo("gold_gradient", with_tagline=True)
        base_np = np.array(base.convert("RGB"))
        # 3 scenes ~3s total
        c1 = ImageClip(base_np).fx(vfx.fadein, 0.6).set_duration(1.0)
        c2 = ImageClip(base_np).set_duration(1.0)
        c3 = ImageClip(base_np).fx(vfx.fadeout, 0.6).set_duration(1.0)
        final = concatenate_videoclips([c1,c2,c3], method="compose")
        mp4 = OUT/"animation/edgeline_intro.mp4"
        final.write_videofile(str(mp4), fps=30, codec="libx264", audio=False, preset="medium", bitrate="3000k", verbose=False, logger=None)
        # quick GIF
        gif = OUT/"animation/edgeline_intro.gif"
        final.write_gif(str(gif), fps=20, program="ffmpeg", fuzz=1)
    except Exception as e:
        # MoviePy not available: skip animation gracefully
        pass

    # Zip everything
    zip_path = OUT/"EdgeLine_Press_Kit.zip"
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file() and p.name != zip_path.name:
                z.write(p, p.relative_to(OUT))

    print(f"✅ Built {zip_path}")

if __name__ == "__main__":
    build_all()
