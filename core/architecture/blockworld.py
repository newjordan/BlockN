"""Write a block-mode build as a Minecraft Java Edition world save.

``to_world(b, path)`` makes a world folder (``level.dat``, ``region/*.mca``
and ``entities/*.mca``) in the 1.21 chunk format, so the build can be opened
in the game, or rendered by a Minecraft path tracer such as Chunky, exactly as
designed. Block entity data (``b.nbt``, kept as SNBT) and entities
(``b.entities``) go in with it. Nothing outside the build is generated: the
rest of the world is empty air.

The writer is self-contained: NBT, SNBT parsing and the Anvil region format
are implemented here.
"""

from __future__ import annotations

import gzip
import io
import math
import os
import struct
import time
import zlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .blockmode import BlockBuild, kind, parse, state

DATA_VERSION = 3955          # Java Edition 1.21.1
MIN_Y, MAX_Y = -64, 320


# ---------------------------------------------------------------- NBT
class Tag:
    """A typed NBT value. ``kind`` is one of: byte short int long float
    double string bytes ints longs list compound."""

    __slots__ = ("kind", "value", "item")

    def __init__(self, kind: str, value: Any, item: Optional[str] = None):
        self.kind, self.value, self.item = kind, value, item

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Tag({self.kind}, {self.value!r})"


_IDS = {"end": 0, "byte": 1, "short": 2, "int": 3, "long": 4, "float": 5, "double": 6, "bytes": 7,
        "string": 8, "list": 9, "compound": 10, "ints": 11, "longs": 12}


def B(v: int) -> Tag: return Tag("byte", v)
def S(v: int) -> Tag: return Tag("short", v)
def I(v: int) -> Tag: return Tag("int", v)
def L(v: int) -> Tag: return Tag("long", v)
def F(v: float) -> Tag: return Tag("float", v)
def D(v: float) -> Tag: return Tag("double", v)
def Str(v: str) -> Tag: return Tag("string", v)
def C(**kv: Tag) -> Tag: return Tag("compound", dict(kv))
def Cd(d: Dict[str, Tag]) -> Tag: return Tag("compound", dict(d))
def Lst(item: str, values: List[Tag]) -> Tag: return Tag("list", list(values), item)
def Ints(v: List[int]) -> Tag: return Tag("ints", list(v))
def Longs(v: List[int]) -> Tag: return Tag("longs", list(v))


def _wstr(out: io.BytesIO, s: str) -> None:
    raw = s.encode("utf-8")
    out.write(struct.pack(">H", len(raw)))
    out.write(raw)


def _payload(out: io.BytesIO, t: Tag) -> None:
    k, v = t.kind, t.value
    if k == "byte":
        out.write(struct.pack(">b", v))
    elif k == "short":
        out.write(struct.pack(">h", v))
    elif k == "int":
        out.write(struct.pack(">i", v))
    elif k == "long":
        out.write(struct.pack(">q", v))
    elif k == "float":
        out.write(struct.pack(">f", v))
    elif k == "double":
        out.write(struct.pack(">d", v))
    elif k == "string":
        _wstr(out, v)
    elif k == "bytes":
        out.write(struct.pack(">i", len(v)))
        out.write(bytes(b & 0xFF for b in v))
    elif k == "ints":
        out.write(struct.pack(">i", len(v)))
        out.write(struct.pack(f">{len(v)}i", *v))
    elif k == "longs":
        out.write(struct.pack(">i", len(v)))
        out.write(struct.pack(f">{len(v)}q", *v))
    elif k == "list":
        item = t.item if v else (t.item or "end")
        out.write(struct.pack(">b", _IDS[item]))
        out.write(struct.pack(">i", len(v)))
        for e in v:
            _payload(out, e)
    elif k == "compound":
        for name, e in v.items():
            out.write(struct.pack(">b", _IDS[e.kind]))
            _wstr(out, name)
            _payload(out, e)
        out.write(b"\x00")
    else:
        raise ValueError(k)


def nbt_bytes(root: Tag, name: str = "") -> bytes:
    out = io.BytesIO()
    out.write(struct.pack(">b", _IDS[root.kind]))
    _wstr(out, name)
    _payload(out, root)
    return out.getvalue()


# ---------------------------------------------------------------- SNBT
class _Snbt:
    """Parses the SNBT the block exporter writes: compounds, lists, quoted and
    bare strings, numbers with b/s/l/f/d suffixes."""

    def __init__(self, text: str):
        self.s, self.i = text, 0

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t\n":
            self.i += 1

    def value(self) -> Tag:
        self.ws()
        c = self.s[self.i]
        if c == "{":
            return self.compound()
        if c == "[":
            return self.list_()
        if c in "\"'":
            return Str(self.quoted())
        return self.scalar(self.bare())

    def quoted(self) -> str:
        q = self.s[self.i]
        self.i += 1
        out = []
        while self.s[self.i] != q:
            if self.s[self.i] == "\\":
                self.i += 1
            out.append(self.s[self.i])
            self.i += 1
        self.i += 1
        return "".join(out)

    def bare(self) -> str:
        j = self.i
        while self.i < len(self.s) and self.s[self.i] not in ",:]}" and self.s[self.i] not in " \t\n":
            self.i += 1
        return self.s[j:self.i]

    @staticmethod
    def scalar(tok: str) -> Tag:
        low = tok.lower()
        try:
            if low in ("true", "false"):
                return B(1 if low == "true" else 0)
            if low[-1:] == "b" and low[:-1].lstrip("-").isdigit():
                return B(int(low[:-1]))
            if low[-1:] == "s" and low[:-1].lstrip("-").isdigit():
                return S(int(low[:-1]))
            if low[-1:] == "l" and low[:-1].lstrip("-").isdigit():
                return L(int(low[:-1]))
            if low[-1:] == "f":
                return F(float(low[:-1]))
            if low[-1:] == "d":
                return D(float(low[:-1]))
            if low.lstrip("-").isdigit():
                return I(int(low))
            if "." in low:
                return D(float(low))
        except ValueError:
            pass
        return Str(tok)

    def compound(self) -> Tag:
        self.i += 1
        d: Dict[str, Tag] = {}
        self.ws()
        while self.s[self.i] != "}":
            self.ws()
            key = self.quoted() if self.s[self.i] in "\"'" else self.bare()
            self.ws()
            assert self.s[self.i] == ":", (self.s, self.i)
            self.i += 1
            d[key] = self.value()
            self.ws()
            if self.s[self.i] == ",":
                self.i += 1
                self.ws()
        self.i += 1
        return Cd(d)

    def list_(self) -> Tag:
        self.i += 1
        self.ws()
        if self.s[self.i:self.i + 2] in ("I;", "L;", "B;"):
            typ = self.s[self.i]
            self.i += 2
            vals = []
            self.ws()
            while self.s[self.i] != "]":
                vals.append(int(self.bare().rstrip("lLbB")))
                self.ws()
                if self.s[self.i] == ",":
                    self.i += 1
                    self.ws()
            self.i += 1
            return {"I": Ints, "L": Longs, "B": lambda v: Tag("bytes", v)}[typ](vals)
        items: List[Tag] = []
        while self.s[self.i] != "]":
            items.append(self.value())
            self.ws()
            if self.s[self.i] == ",":
                self.i += 1
                self.ws()
        self.i += 1
        return Lst(items[0].kind if items else "end", items)


def snbt(text: str) -> Tag:
    return _Snbt(text).value()


# ---------------------------------------------------------------- chunks
_BLOCK_ENTITY_IDS = {
    "chest": "chest", "trapped_chest": "trapped_chest", "barrel": "barrel", "furnace": "furnace",
    "blast_furnace": "blast_furnace", "smoker": "smoker", "lectern": "lectern", "bell": "bell",
    "decorated_pot": "decorated_pot", "chiseled_bookshelf": "chiseled_bookshelf", "brewing_stand": "brewing_stand",
}


def _block_entity_id(name: str) -> Optional[str]:
    if name in _BLOCK_ENTITY_IDS:
        return _BLOCK_ENTITY_IDS[name]
    if name.endswith("_banner"):
        return "banner"
    if name.endswith("_sign"):
        return "sign"
    if name.endswith("_bed"):
        return "bed"
    if name == "campfire" or name == "soul_campfire":
        return name
    return None


def _state_tag(block: str) -> Tag:
    name, props = parse(block)
    d: Dict[str, Tag] = {"Name": Str("minecraft:" + name)}
    if props:
        d["Properties"] = Cd({k: Str(v) for k, v in sorted(props.items())})
    return Cd(d)


def _pack(indices: List[int], bits: int) -> List[int]:
    per = 64 // bits
    out = []
    for j in range(0, 4096, per):
        v = 0
        for k, idx in enumerate(indices[j:j + per]):
            v |= idx << (k * bits)
        if v >= 1 << 63:
            v -= 1 << 64
        out.append(v)
    return out


def _entity_tag(e: Dict[str, Any], dy: int) -> Tag:
    from .blockmode import _FACING_2D, _FACING_3D, snbt_items
    x, y, z = e["pos"]
    y += dy
    eid = e["id"] if ":" in e["id"] else "minecraft:" + e["id"]
    name = eid.split(":")[-1]
    d: Dict[str, Tag] = {"id": Str(eid), "Motion": Lst("double", [D(0), D(0), D(0)]),
                         "Rotation": Lst("float", [F(0), F(0)]), "OnGround": B(0), "Invulnerable": B(0)}
    if name in ("item_frame", "glow_item_frame"):
        f = _FACING_3D[e["facing"]]
        off = {2: (0.5, 0.5, 0.97), 3: (0.5, 0.5, 0.03), 4: (0.97, 0.5, 0.5), 5: (0.03, 0.5, 0.5), 1: (.5, .03, .5),
               0: (.5, .97, .5)}[f]
        item = e.get("item", "air")
        d.update(Pos=Lst("double", [D(x + off[0]), D(y + off[1]), D(z + off[2])]), Facing=B(f), Fixed=B(1),
                 TileX=I(x), TileY=I(y), TileZ=I(z),
                 Item=C(id=Str(item if ":" in item else "minecraft:" + item), count=I(1)))
    elif name == "painting":
        f = _FACING_2D[e["facing"]]
        off = {0: (0.5, 0.5, 0.03), 1: (0.97, 0.5, 0.5), 2: (0.5, 0.5, 0.97), 3: (0.03, 0.5, 0.5)}[f]
        var = e.get("variant", "kebab")
        d.update(Pos=Lst("double", [D(x + off[0]), D(y + off[1]), D(z + off[2])]), facing=B(f),
                 variant=Str(var if ":" in var else "minecraft:" + var), TileX=I(x), TileY=I(y), TileZ=I(z))
    else:
        d.update(Pos=Lst("double", [D(x + .5), D(y + .0625), D(z + .5)]))
        if e.get("items"):
            d["Items"] = snbt(snbt_items(e["items"]))
    return Cd(d)


def _region_file(path: str, chunks: Dict[Tuple[int, int], bytes]) -> None:
    """Write an Anvil region file from {(cx & 31, cz & 31): uncompressed NBT}."""
    offsets = bytearray(4096)
    stamps = bytearray(4096)
    body = io.BytesIO()
    sector = 2
    now = int(time.time())
    for (lx, lz), raw in sorted(chunks.items()):
        data = zlib.compress(raw)
        blob = struct.pack(">ib", len(data) + 1, 2) + data
        n = (len(blob) + 4095) // 4096
        blob += b"\x00" * (n * 4096 - len(blob))
        i = 4 * (lx + lz * 32)
        offsets[i:i + 4] = struct.pack(">I", (sector << 8) | n)
        stamps[i:i + 4] = struct.pack(">I", now)
        body.write(blob)
        sector += n
    with open(path, "wb") as f:
        f.write(offsets)
        f.write(stamps)
        f.write(body.getvalue())


def to_world(b: BlockBuild, path: str, name: Optional[str] = None, dy: int = 0, ground: bool = True,
             biome: str = "minecraft:plains", fill: Optional[str] = None) -> Dict[str, Any]:
    """Write ``b`` as a world folder at ``path``; block (x, y, z) of the build
    lands at world (x, y + dy, z). Returns a summary.

    ``fill`` (a block such as "stone") packs the ground under the build's
    footprint with that block, down to two below its lowest block, so a
    build that digs below the ground (a mine) sits in solid rock rather
    than hanging over the void of a flat world."""
    blocks: Dict[Tuple[int, int, int], str] = {}
    if fill and b.blocks:
        xs0 = [p[0] for p in b.blocks]; zs0 = [p[2] for p in b.blocks]
        lo = min(min(p[1] for p in b.blocks), 0) - 2
        rock = state(fill) if ":" not in fill else fill
        for x in range(min(xs0) - 2, max(xs0) + 3):
            for z in range(min(zs0) - 2, max(zs0) + 3):
                for y in range(lo, 0):
                    blocks[(x, y + dy, z)] = rock
    if ground:
        for (x, z), blk in b.ground.items():
            blocks[(x, -1 + dy, z)] = blk
    for (x, y, z), blk in b.blocks.items():
        blocks[(x, y + dy, z)] = blk
    ys = [p[1] for p in blocks]
    if ys and (min(ys) < MIN_Y or max(ys) >= MAX_Y):
        raise ValueError(f"build spans y {min(ys)}..{max(ys)}; shift it with dy into {MIN_Y}..{MAX_Y - 1}")
    # Group by chunk section.
    sections: Dict[Tuple[int, int], Dict[int, Dict[int, str]]] = {}
    for (x, y, z), blk in blocks.items():
        cx, cz, sy = x >> 4, z >> 4, y >> 4
        sections.setdefault((cx, cz), {}).setdefault(sy, {})[((y & 15) << 8) | ((z & 15) << 4) | (x & 15)] = blk
    bents: Dict[Tuple[int, int], List[Tag]] = {}
    for (x, y, z), blk in b.blocks.items():
        bid = _block_entity_id(parse(blk)[0])
        data = b.nbt.get((x, y, z))
        if bid is None and data is None:
            continue
        d: Dict[str, Tag] = {}
        if data:
            d.update(snbt(data).value)
        d.update(id=Str("minecraft:" + (bid or parse(blk)[0])), x=I(x), y=I(y + dy), z=I(z), keepPacked=B(0))
        bents.setdefault((x >> 4, z >> 4), []).append(Cd(d))
    ents: Dict[Tuple[int, int], List[Tag]] = {}
    for e in b.entities:
        x, _, z = e["pos"]
        ents.setdefault((x >> 4, z >> 4), []).append(_entity_tag(e, dy))

    os.makedirs(os.path.join(path, "region"), exist_ok=True)
    os.makedirs(os.path.join(path, "entities"), exist_ok=True)
    regions: Dict[Tuple[int, int], Dict[Tuple[int, int], bytes]] = {}
    eregions: Dict[Tuple[int, int], Dict[Tuple[int, int], bytes]] = {}
    air = "minecraft:air"
    lo_s, hi_s = MIN_Y >> 4, (MAX_Y - 1) >> 4
    for (cx, cz), secs in sections.items():
        sec_tags = []
        for sy in range(lo_s, hi_s + 1):
            cells = secs.get(sy)
            if not cells:
                bs = C(palette=Lst("compound", [_state_tag(air)]))
            else:
                palette = [air] + sorted(set(cells.values()))
                index = {blk: i for i, blk in enumerate(palette)}
                idx = [0] * 4096
                for k, blk in cells.items():
                    idx[k] = index[blk]
                bits = max(4, math.ceil(math.log2(len(palette))))
                bs = C(palette=Lst("compound", [_state_tag(p) for p in palette]), data=Longs(_pack(idx, bits)))
            sec_tags.append(C(Y=B(sy), block_states=bs, biomes=C(palette=Lst("string", [Str(biome)])),
                              BlockLight=Tag("bytes", [0] * 2048), SkyLight=Tag("bytes", [-1] * 2048)))
        chunk = C(DataVersion=I(DATA_VERSION), xPos=I(cx), zPos=I(cz), yPos=I(lo_s), Status=Str("minecraft:full"),
                  LastUpdate=L(0), InhabitedTime=L(0), isLightOn=B(0),
                  sections=Lst("compound", sec_tags),
                  block_entities=Lst("compound", bents.get((cx, cz), [])),
                  Heightmaps=C(), structures=C(References=C(), starts=C()),
                  PostProcessing=Lst("list", []))
        regions.setdefault((cx >> 5, cz >> 5), {})[(cx & 31, cz & 31)] = nbt_bytes(chunk)
    for (cx, cz), lst in ents.items():
        echunk = C(DataVersion=I(DATA_VERSION), Position=Ints([cx, cz]), Entities=Lst("compound", lst))
        eregions.setdefault((cx >> 5, cz >> 5), {})[(cx & 31, cz & 31)] = nbt_bytes(echunk)
    for (rx, rz), chunks in regions.items():
        _region_file(os.path.join(path, "region", f"r.{rx}.{rz}.mca"), chunks)
    for (rx, rz), chunks in eregions.items():
        _region_file(os.path.join(path, "entities", f"r.{rx}.{rz}.mca"), chunks)

    xs = [p[0] for p in blocks] or [0]
    zs = [p[2] for p in blocks] or [0]
    spawn = (int(sum(xs) / len(xs)), max(ys or [64]) + 1, int(sum(zs) / len(zs)))
    level = C(Data=C(
        LevelName=Str(name or b.product), DataVersion=I(DATA_VERSION), version=I(19133), GameType=I(1),
        SpawnX=I(spawn[0]), SpawnY=I(spawn[1]), SpawnZ=I(spawn[2]), Time=L(6000), DayTime=L(6000),
        LastPlayed=L(int(time.time() * 1000)), allowCommands=B(1), hardcore=B(0), Difficulty=B(0),
        initialized=B(1), raining=B(0), thundering=B(0),
        Version=C(Id=I(DATA_VERSION), Name=Str("1.21.1"), Series=Str("main"), Snapshot=B(0)),
        WorldGenSettings=C(seed=L(0), generate_features=B(0), bonus_chest=B(0), dimensions=C(
            **{"minecraft:overworld": C(type=Str("minecraft:overworld"), generator=C(
                type=Str("minecraft:flat"), settings=C(layers=Lst("compound", []), biome=Str(biome),
                                                       features=B(0), lakes=B(0), structure_overrides=Lst("end", []))))})),
        GameRules=C(doDaylightCycle=Str("false"), doMobSpawning=Str("false"), doWeatherCycle=Str("false")),
    ))
    with gzip.open(os.path.join(path, "level.dat"), "wb") as f:
        f.write(nbt_bytes(level))
    return {"chunks": len(sections), "regions": len(regions), "block_entities": sum(map(len, bents.values())),
            "entities": len(b.entities), "spawn": spawn, "y_range": (min(ys or [0]), max(ys or [0]))}


# ---------------------------------------------------------------- reading back
def read_nbt(raw: bytes) -> Tag:
    """Read NBT bytes back (for tests and checks)."""
    f = io.BytesIO(raw)

    def rd(fmt: str) -> Any:
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr() -> str:
        n = rd(">H")
        return f.read(n).decode("utf-8")

    inv = {v: k for k, v in _IDS.items()}

    def payload(tid: int) -> Tag:
        k = inv[tid]
        if k == "byte":
            return B(rd(">b"))
        if k == "short":
            return S(rd(">h"))
        if k == "int":
            return I(rd(">i"))
        if k == "long":
            return L(rd(">q"))
        if k == "float":
            return F(rd(">f"))
        if k == "double":
            return D(rd(">d"))
        if k == "string":
            return Str(rstr())
        if k == "bytes":
            n = rd(">i")
            return Tag("bytes", list(struct.unpack(f">{n}b", f.read(n))))
        if k == "ints":
            n = rd(">i")
            return Ints(list(struct.unpack(f">{n}i", f.read(4 * n))))
        if k == "longs":
            n = rd(">i")
            return Longs(list(struct.unpack(f">{n}q", f.read(8 * n))))
        if k == "list":
            it, n = rd(">b"), rd(">i")
            return Lst(inv[it], [payload(it) for _ in range(n)])
        d = {}
        while True:
            t = rd(">b")
            if t == 0:
                return Cd(d)
            nm = rstr()
            d[nm] = payload(t)

    tid = rd(">b")
    rstr()
    return payload(tid)


def read_chunk(region_path: str, lx: int, lz: int) -> Optional[Tag]:
    with open(region_path, "rb") as f:
        head = f.read(4096)
        loc = struct.unpack(">I", head[4 * (lx + lz * 32):4 * (lx + lz * 32) + 4])[0]
        if loc == 0:
            return None
        f.seek((loc >> 8) * 4096)
        n, comp = struct.unpack(">ib", f.read(5))
        return read_nbt(zlib.decompress(f.read(n - 1)))


def block_at(chunk: Tag, x: int, y: int, z: int) -> str:
    """The block state name at world (x, y, z) in a chunk read back."""
    for sec in chunk.value["sections"].value:
        if sec.value["Y"].value != y >> 4:
            continue
        bs = sec.value["block_states"].value
        pal = bs["palette"].value
        if "data" not in bs:
            return pal[0].value["Name"].value
        bits = max(4, math.ceil(math.log2(len(pal))))
        per = 64 // bits
        k = ((y & 15) << 8) | ((z & 15) << 4) | (x & 15)
        v = bs["data"].value[k // per] & ((1 << 64) - 1)
        i = (v >> ((k % per) * bits)) & ((1 << bits) - 1)
        return pal[i].value["Name"].value
    return "minecraft:air"


__all__ = ["to_world", "snbt", "nbt_bytes", "read_nbt", "read_chunk", "block_at"]
