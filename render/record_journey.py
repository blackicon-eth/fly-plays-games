import sys, collections, numpy as np, random, time, argparse
sys.path.insert(0, "."); sys.path.insert(0, "fly-ai")
from pokesim import PokemonAdapter
from flybrain import FlyBrain

ap = argparse.ArgumentParser(description="Record the fly journey (Pallet Town -> Viridian City) to render/recording.npz.")
ap.add_argument("--window", default="null", help="PyBoy window: 'null' (fast, default) or 'SDL2' to watch it live")
ap.add_argument("--scale", type=int, default=3, help="PyBoy window scale when --window SDL2")
args = ap.parse_args()

WALKSET = {0x00,0x10,0x1b,0x20,0x21,0x23,0x2c,0x2d,0x2e,0x30,0x31,0x33,0x39,0x3c,0x3e,0x52,0x54,0x58,0x5b}
LEDGE = {0x36,0x37,0x27,0x0D,0x1D}
FACE = {"up":4,"down":0,"left":8,"right":12}
NAME = {(0,-1):"up",(0,1):"down",(-1,0):"left",(1,0):"right",(0,2):"down"}
DIRIDX = {"up":0,"down":1,"left":2,"right":3}
BTNIDX = {"up":0,"down":1,"left":2,"right":3,"a":4,"b":5}
def walk(t): return (t is not None) and (t in WALKSET) and (t not in LEDGE)

b = FlyBrain(data="data/fly-data", device="cpu"); az = b.azimuth; src = np.linspace(-1, 1, 160); prev = None
a = PokemonAdapter(window=args.window, scale=args.scale)
def pos(): return (a.read("x"), a.read("y"))
def tm(): return [[a.read(0xC3A0 + r*20 + c) for c in range(20)] for r in range(18)]
FR, F, DIRS, PXY, BTN, MAPS = [], [], [], [], [], []
CAP = 20000

def rec(name):
    global prev
    rgb = a.screen(gray=False)
    g = rgb.astype(np.float32) @ np.array([0.299,0.587,0.114], np.float32)
    band = g[40:104].mean(0) / 255.0
    up = np.interp(az, src, band)
    ch = np.abs(up - prev) if prev is not None else np.zeros_like(up); prev = up
    dr = np.clip(0.45*up + 1.6*ch, 0, 1).astype(np.float32)
    f = b.step(eye_drive=dr)
    FR.append(rgb); F.append(f.astype(np.int32)); DIRS.append(DIRIDX.get(name,3))
    PXY.append(pos()); BTN.append(BTNIDX.get(name,6)); MAPS.append(a.read("map"))

def tap(d):
    for _ in range(3):
        if a.read(0xC109) == FACE[d]: break
        a.pb.button_press(d); a.step(1); a.pb.button_release(d)

def pr(btn, k):
    a.pb.button_press(btn)
    for _ in range(k): a.step(1); rec(btn)
    a.pb.button_release(btn)

def flee():
    for _ in range(14):
        if not a.read("in_battle"): return
        pr("a",6); pr("b",4); pr("down",6); pr("right",6); pr("a",6)
        for _ in range(4):
            if not a.read("in_battle"): return
            pr("a",6)

def move(d):
    tap(d); x0, y0 = pos(); m0 = a.read("map")
    a.pb.button_press(d)
    for _ in range(16):
        a.step(1)
        if len(FR) < CAP: rec(d)
    a.pb.button_release(d)
    if a.read("in_battle"):
        if a.read(0xD057) == 1: flee()
        else:
            n = 0
            while a.read("in_battle") and n < 400: pr("a", 8); n += 1
    return (pos() != (x0, y0)) or (a.read("map") != m0)

# ---- opening: walk around Pallet Town (shows the starting point) ----
a.load_scene("pallet")
print("pallet start", pos(), flush=True)
for _ in range(3): move("down")
for _ in range(5): move("right")
for _ in range(10): move("left")
for _ in range(5): move("right")
for _ in range(3): move("up")
print("pallet walk done, frames", len(FR), "pos", pos(), flush=True)

# ---- cut to the Route 1 border and navigate to Viridian ----
a.load_scene("route1")
print("route1 start", pos(), flush=True)
known = {}; blocked = set(); t0 = time.time(); reached = False
for it in range(600):
    m = a.read("map")
    if m == 1: reached = True; break
    if len(FR) >= CAP: print("CAP reached", flush=True); break
    if m != 12:
        for _ in range(20):
            move("down")
            if a.read("map") == 12: break
        continue
    G = tm(); px, py = pos()
    for r in range(18):
        for c in range(20):
            known[(m, px + (c-8), py + (r-8))] = G[r][c]

    def neigh(x, y):
        out = []; mm = m
        t = known.get((mm, x, y-1))
        if t is None: out.append((x, y-1))
        elif walk(t): out.append((x, y-1))
        for dx in (-1, 1):
            t = known.get((mm, x+dx, y))
            if t is None: out.append((x+dx, y))
            elif walk(t): out.append((x+dx, y))
        t = known.get((mm, x, y+1))
        if t is None: out.append((x, y+1))
        elif t in LEDGE:
            t2 = known.get((mm, x, y+2))
            if t2 is not None and walk(t2): out.append((x, y+2))
        elif walk(t): out.append((x, y+1))
        return out

    cand = [(X, Y) for (mm, X, Y) in known if mm == m and walk(known.get((mm, X, Y))) and any((mm, X+dd[0], Y+dd[1]) not in known for dd in [(0,-1),(0,1),(-1,0),(1,0)])]
    start = (px, py); pv = {start: None}; q = collections.deque([start])
    while q:
        cc = q.popleft()
        for nb in neigh(*cc):
            if walk(known.get((m, nb[0], nb[1]))) and (m, nb[0], nb[1]) not in blocked and nb not in pv:
                pv[nb] = cc; q.append(nb)
    reach = [c for c in cand if c in pv]
    if not reach:
        blocked.clear(); rng = random.Random(it)
        for _ in range(30):
            move(rng.choice(["up","up","left","right","down"]))
            if a.read("map") == 1: reached = True; break
        if reached: break
        continue
    tgt = min(reach, key=lambda c: (c[1] + abs(c[0]-px)))
    path = []; c = tgt
    while pv.get(c):
        p = pv[c]; path.append((c[0]-p[0], c[1]-p[1])); c = p
    path = path[::-1]; dx, dy = path[0]; nm = NAME.get((dx, dy))
    if nm is None: blocked.add((m, px+dx, py+dy)); continue
    if not move(nm): blocked.add((m, px+dx, py+dy)); known[(m, px+dx, py+dy)] = 0x99
    if it % 20 == 0: print("it", it, "map", a.read("map"), "pos", pos(), "frames", len(FR), flush=True)

a.close()
lens = np.array([len(f) for f in F])
np.savez("render/recording.npz", frames=np.array(FR, np.uint8), fired=np.concatenate(F),
         starts=np.concatenate([[0], np.cumsum(lens)]), dirs=np.array(DIRS, np.int8),
         pxy=np.array(PXY, np.int16), btns=np.array(BTN, np.int8), maps=np.array(MAPS, np.int16))
print("reached", reached, "frames", len(FR), "in %.1fs" % (time.time()-t0))
