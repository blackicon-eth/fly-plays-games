import sys, subprocess, math, numpy as np, time, argparse
from pathlib import Path
GAME = Path(__file__).resolve().parents[1]      # pokemon-red/
ROOT = GAME.parent                              # repository root (data/, fly-ai/)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "fly-ai"))
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from matplotlib import colormaps
from flybrain import FlyBrain

ap = argparse.ArgumentParser(description="Render a recorded journey to mp4.")
ap.add_argument("--input", default=str(GAME / "render" / "recording.npz"))
ap.add_argument("--output", default=str(GAME / "media" / "viridian_final.mp4"))
ap.add_argument("--caption", default="fly taps the buttons the game receives")
ap.add_argument("--bar-label", default="journey to Viridian")
ap.add_argument("--max-frames", type=int, default=2625)
args = ap.parse_args()

d = np.load(args.input)
frames = d["frames"]; N = len(frames)
fired = d["fired"].astype(np.int64); starts = d["starts"]; btns = d["btns"]; maps = d["maps"]
tb = int(np.argmax(maps == 12)) if (maps == 12).any() else -1
b = FlyBrain(data=str(ROOT / "data" / "fly-data"), device="cpu"); n = b.n; sc = b.superclass.astype(str); side = b.side.astype(str)
label = np.zeros(n, np.uint8)
label[sc == "visual_projection"] = 1
label[(sc == "descending_neuron") & (side == "L")] = 2
label[(sc == "descending_neuron") & (side == "R")] = 3
cnt = np.zeros((N, 4), np.int32)
for t in range(N):
    seg = fired[starts[t]:starts[t+1]]
    if len(seg): cnt[t] = np.bincount(label[seg], minlength=4)
P = b.positions.copy(); fin = np.isfinite(P).all(1); xy = np.full((n, 2), np.nan)
Pf = P[fin]; xy[fin] = Pf @ np.linalg.svd(Pf - Pf.mean(0), full_matrices=False)[2][:2].T
xmin, ymin = np.nanmin(xy, 0); xmax, ymax = np.nanmax(xy, 0); xf, yf = xy[fin, 0], xy[fin, 1]
lut = (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
W = 15
def win_all(t):
    lo = max(0, t - W); seg = fired[starts[lo]:starts[t]]
    return np.bincount(seg, minlength=n) if len(seg) else np.zeros(n, np.int32)
def brain_img(t, w, h):
    H, _, _ = np.histogram2d(xf, yf, bins=(160, 180), range=[[xmin, xmax], [ymin, ymax]], weights=win_all(t)[fin])
    v = np.log1p(H.T); v = np.clip(v / (np.log1p(H.max()) + 1e-9) * 1.7, 0, 1)
    return Image.fromarray(lut[(v * 255).astype(np.uint8)]).transpose(Image.FLIP_TOP_BOTTOM).resize((w, h), Image.BILINEAR)

BUT = {"up":(190,506),"down":(190,612),"left":(134,559),"right":(246,559),"a":(312,532),"b":(346,590)}
def draw_pad(dr, active, font):
    for name,(bx,by) in BUT.items():
        on = (active == name)
        r = 26 if name in ("up","down","left","right") else 22
        fill = (255, 215, 100); edge = (255, 235, 150)
        if not on: fill = (34, 38, 52); edge = (90, 96, 120)
        if on: r += 3
        if name in ("up","down","left","right"):
            dr.polygon([(bx-r,by),(bx,by-r),(bx+r,by),(bx,by+r)], fill=fill, outline=edge)
        else:
            dr.ellipse([bx-r,by-r,bx+r,by+r], fill=fill, outline=edge)
        lbl = {"up":"^","down":"v","left":"<","right":">","a":"A","b":"B"}[name]
        dr.text((bx-4,by-8), lbl, font=font, fill=(20,22,30) if on else (170,176,196))

def draw_fly(dr, cx, cy, s, phase, flap, press):
    ang = math.pi/2; ca, sa = math.cos(ang), math.sin(ang)
    def R(px, py): return (cx + (px*ca - py*sa)*s, cy + (px*sa + py*ca)*s)
    for lx, sd in [(-0.7,-1),(-0.7,1),(-0.1,-1),(-0.1,1),(0.6,-1),(0.6,1)]:
        root = R(lx, 0.25*sd); knee = R(lx+0.3, 1.1*sd); foot = R(lx+0.9, 1.9*sd)
        dr.line([root, knee, foot], fill=(150,150,175,235), width=max(1,int(0.14*s)))
    if press is not None:
        root = R(1.0, -0.35); knee = ((root[0]+press[0])/2-16, (root[1]+press[1])/2-6)
        dr.line([root, knee, press], fill=(205,210,230,255), width=max(1,int(0.15*s)))
        dr.ellipse([press[0]-5,press[1]-5,press[0]+5,press[1]+5], fill=(255,240,180,255))
    ab = [R(x,y) for x,y in [(-0.2,-0.6),(-2.2,-0.5),(-2.6,0),(-2.2,0.5),(-0.2,0.6)]]
    dr.polygon(ab, fill=(72,76,94,255), outline=(132,137,162,255))
    th = [R(x,y) for x,y in [(1.1,-0.7),(0.4,-0.9),(-0.3,-0.7),(-0.3,0.7),(0.4,0.9),(1.1,0.7)]]
    dr.polygon(th, fill=(86,90,110,255), outline=(154,159,184,255))
    hx, hy = R(1.35, 0); dr.ellipse([hx-0.55*s, hy-0.5*s, hx+0.55*s, hy+0.5*s], fill=(62,66,84,255), outline=(142,147,172,255))
    for eo in (-0.28, 0.28):
        ex, ey = R(1.5, eo); dr.ellipse([ex-0.22*s, ey-0.22*s, ex+0.22*s, ey+0.22*s], fill=(205,60,70,255))
    f = (0.12 + 0.18*flap) * math.sin(phase * 2.2)
    for sd in (-1, 1):
        wpts = [(-0.1,0.30*sd),(-1.5,(0.55+0.35*abs(f))*sd),(-3.0,(1.15+0.5*abs(f))*sd),(-1.0,(0.95+0.4*abs(f))*sd)]
        poly = [R(x, y) for x,y in wpts]
        dr.polygon(poly, fill=(120,190,235,85), outline=(180,220,255,190))

def bar(dr, x, y, w, h, frac, col):
    dr.rectangle([x,y,x+w,y+h], outline=(90,95,120)); dr.rectangle([x,y,x+w*max(0,min(1,frac)),y+h], fill=col)

CW, CH = 1280, 720
try:
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 15)
    fontS = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
    fontB = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception:
    font = fontS = fontB = ImageFont.load_default()

idxs = np.linspace(0, N - 1, min(N, args.max_frames)).astype(int).tolist()
print("recorded", N, "-> frames", len(idxs), "-> %.1fs at 25fps" % (len(idxs)/25), flush=True)
proc = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{CW}x{CH}",
                         "-r","25","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-b:v","6000k",
                         args.output], stdin=subprocess.PIPE)
vpr = cnt[:,1]; dn = cnt[:,2]+cnt[:,3]; fl = cnt[:,1]
def smooth(a,k=40): return np.convolve(a.astype(np.float32), np.ones(k,np.float32)/k, mode="same")
fln = smooth(fl); fln = fln/(fln.max()+1e-6)
NAMES = {0:"Pallet Town", 12:"Route 1", 1:"Viridian City"}
t0=time.time()
for k,t in enumerate(idxs):
    cv = Image.new("RGB",(CW,CH),(6,7,11)); dr = ImageDraw.Draw(cv,"RGBA")
    dr.rectangle([0,0,380,CH], fill=(9,11,16))
    active = {0:"up",1:"down",2:"left",3:"right",4:"a",5:"b"}.get(int(btns[t]))
    draw_pad(dr, active, fontB)
    press = BUT[active] if active else None
    draw_fly(dr, 190, 372, 30, 0.15*t, fln[t], press)
    dr.text((12,12), args.caption, font=fontS, fill=(170,180,200))
    dr.text((12, CH-24), ("pressed: "+active.upper()) if active else "pressed: -", font=font, fill=(255,220,120))
    gx, gy, gw, gh = 400, 40, 368, 331
    dr.text((gx,14), "Pokémon Red  |  %s" % NAMES.get(int(maps[t]), "?"), font=font, fill=(230,230,240))
    cv.paste(Image.fromarray(frames[t]).resize((gw,gh),Image.NEAREST),(gx,gy))
    bar(dr,gx,gy+gh+14,368,10,t/N,(120,220,140)); dr.text((gx,gy+gh+28),"%s %.0f%%"%(args.bar_label,100*t/N),font=fontS,fill=(120,220,140))
    ty=gy+gh+60; th2=150
    dr.rectangle([gx,ty,gx+368,ty+th2],outline=(40,44,60))
    for arr,col in [(vpr,(57,208,255)),(dn,(255,77,109))]:
        seg = arr[max(0,t-200):t+1]; mx = max(1, seg.max() if len(seg) else 1)
        pts=[(gx+(j-max(0,t-200))/200*368, ty+th2-arr[j]/mx*(th2-10)) for j in range(max(0,t-200),t)]
        if len(pts)>1: dr.line(pts,fill=col,width=1)
    dr.text((gx+6,ty+6),"visual_projection  /  descending",font=fontS,fill=(180,180,200))
    asp=(xmax-xmin)/(ymax-ymin); bh=520; bw=min(440,int(bh*asp)); bx=790+(480-bw)//2; by=80
    dr.text((bx,50),"MaleCNS v1.0  |  166,700 neurons",font=font,fill=(230,230,240))
    cv.paste(brain_img(t,bw,bh),(bx,by+24))
    if tb >= 0:
        fade = min(1.0, abs(t - tb) / 30.0)
        if fade < 1.0: cv = ImageEnhance.Brightness(cv).enhance(fade)
    proc.stdin.write(cv.tobytes())
    if k%300==0: print("frame",k,"/",len(idxs),"%.1fs"%(time.time()-t0),flush=True)
proc.stdin.close(); proc.wait()
print("done %.1fs"%(time.time()-t0))
