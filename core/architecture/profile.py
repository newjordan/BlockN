"""Player and camera profiles: the body and eye a build is made for.

Architecture authored for people uses fixed human sizes: a Victorian door is
0.84 by 2.05 metres whoever walks through it. A game character may be 1.2 or
2.6 metres tall, run at 3 or 12 metres a second and jump 1 or 4 metres. A
``PlayerProfile`` carries those numbers, and ``Clearances`` derives from them
the sizes an architect should build to and the voxel checks should hold it to.

The profile is optional on a request. Without one nothing changes, and a
document keeps the fingerprint it had before profiles existed.

Units are metres, seconds and degrees. Y is up, as everywhere in BlockN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Mapping, Optional

PROFILE_SCHEMA = "blockn.player/v1"
CAMERA_MODES = ("first_person", "third_person", "top_down", "isometric", "side")

# Clearance added each side of the body in a doorway or corridor, and above the
# head in a doorway. These are design margins, not physics.
SIDE_CLEARANCE = 0.15
HEAD_CLEARANCE = 0.30
# A reaction time and a braking deceleration for sight-distance checks.
DEFAULT_REACTION_S = 0.35
DEFAULT_BRAKE = 20.0


def _number(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"profile {name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"profile {name} must be a number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"profile {name} must be between {minimum} and {maximum}")
    return result


@dataclass(frozen=True)
class Camera:
    mode: str = "third_person"
    fov_deg: float = 70.0
    boom: float = 3.5           # third person: pivot-to-camera distance
    height: float = 1.6         # camera pivot height above the feet
    pitch_deg: float = -15.0    # negative looks down

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Camera":
        data = dict(data or {})
        unknown = set(data) - {"mode", "fov_deg", "boom", "height", "pitch_deg"}
        if unknown:
            raise ValueError(f"unknown camera field(s): {', '.join(sorted(unknown))}")
        mode = str(data.get("mode", cls.mode)).strip().lower()
        if mode not in CAMERA_MODES:
            raise ValueError(f"camera mode must be one of: {', '.join(CAMERA_MODES)}")
        return cls(
            mode=mode,
            fov_deg=_number("camera.fov_deg", data.get("fov_deg", cls.fov_deg), 10.0, 170.0),
            boom=_number("camera.boom", data.get("boom", cls.boom), 0.0, 50.0),
            height=_number("camera.height", data.get("height", cls.height), 0.0, 20.0),
            pitch_deg=_number("camera.pitch_deg", data.get("pitch_deg", cls.pitch_deg), -90.0, 90.0),
        )


@dataclass(frozen=True)
class PlayerProfile:
    """The body, movement and camera a level is built for."""

    name: str = "human"
    radius: float = 0.4          # capsule radius
    height: float = 1.8          # capsule full height
    walk_speed: float = 3.0
    run_speed: float = 7.0
    jump_height: float = 2.4     # apex height of a standing jump
    gravity: float = 20.0
    step_height: float = 0.35    # tallest riser walked up without jumping
    max_drop: float = 3.0        # tallest drop taken without harm
    max_slope_deg: float = 45.0
    reaction_s: float = DEFAULT_REACTION_S
    camera: Camera = field(default_factory=Camera)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str | None) -> "PlayerProfile":
        """Build from a dict, or from a preset name such as ``"giant"``.

        A dict may name a ``preset`` and override any field of it.
        """
        if data is None:
            return cls()
        if isinstance(data, str):
            return preset(data)
        if not isinstance(data, Mapping):
            raise ValueError("profile must be a preset name or an object")
        data = dict(data)
        data.pop("schema", None)
        base = preset(data.pop("preset")) if "preset" in data else cls()
        allowed = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown profile field(s): {', '.join(sorted(unknown))}")
        merged = {**_plain(base), **data}
        camera = Camera.from_dict({**asdict(base.camera), **dict(data.get("camera") or {})})
        profile = cls(
            name=str(merged["name"]).strip().lower() or "custom",
            radius=_number("radius", merged["radius"], 0.05, 10.0),
            height=_number("height", merged["height"], 0.2, 30.0),
            walk_speed=_number("walk_speed", merged["walk_speed"], 0.1, 100.0),
            run_speed=_number("run_speed", merged["run_speed"], 0.1, 200.0),
            jump_height=_number("jump_height", merged["jump_height"], 0.0, 50.0),
            gravity=_number("gravity", merged["gravity"], 0.1, 200.0),
            step_height=_number("step_height", merged["step_height"], 0.01, 5.0),
            max_drop=_number("max_drop", merged["max_drop"], 0.0, 100.0),
            max_slope_deg=_number("max_slope_deg", merged["max_slope_deg"], 0.0, 89.0),
            reaction_s=_number("reaction_s", merged["reaction_s"], 0.0, 5.0),
            camera=camera,
        )
        if profile.radius * 2 > profile.height:
            raise ValueError("profile radius must be at most half the height")
        return profile

    def to_dict(self) -> Dict[str, Any]:
        out = _plain(self)
        out["camera"] = asdict(self.camera)
        out["schema"] = PROFILE_SCHEMA
        return {key: out[key] for key in sorted(out)}

    def clearances(self) -> "Clearances":
        return Clearances.of(self)


def _plain(profile: PlayerProfile) -> Dict[str, Any]:
    out = asdict(profile)
    out["camera"] = asdict(profile.camera)
    return out


PRESETS: Dict[str, PlayerProfile] = {
    "human": PlayerProfile(),
    # A child-sized or small-creature character.
    "small": PlayerProfile(name="small", radius=0.25, height=1.1, walk_speed=2.5, run_speed=6.0,
                           jump_height=1.4, step_height=0.22, max_drop=2.0,
                           camera=Camera(height=0.95, boom=2.6)),
    # A 2.4 m brute: wider, taller, slower, a short jump.
    "brute": PlayerProfile(name="brute", radius=0.6, height=2.4, walk_speed=2.5, run_speed=5.0,
                           jump_height=1.2, step_height=0.45, max_drop=3.5,
                           camera=Camera(height=2.2, boom=4.5)),
    # A 2.6 m giant, the "8-foot character" case.
    "giant": PlayerProfile(name="giant", radius=0.7, height=2.6, walk_speed=2.2, run_speed=4.5,
                           jump_height=1.0, step_height=0.5, max_drop=4.0,
                           camera=Camera(height=2.4, boom=5.0)),
    # A fast, high-jumping action character.
    "runner": PlayerProfile(name="runner", radius=0.35, height=1.8, walk_speed=4.0, run_speed=12.0,
                            jump_height=3.2, step_height=0.4, max_drop=6.0),
    # A Minecraft-like player in a 1-metre block world: 0.6 m wide, 1.8 m tall,
    # steps one 0.5 m slab, jumps a little over one block.
    "blocky": PlayerProfile(name="blocky", radius=0.3, height=1.8, walk_speed=4.3, run_speed=5.6,
                            jump_height=1.25, gravity=32.0, step_height=0.6, max_drop=3.0,
                            camera=Camera(mode="first_person", height=1.62, boom=0.0, pitch_deg=0.0)),
}


def preset(name: str) -> PlayerProfile:
    key = str(name).strip().lower()
    if key not in PRESETS:
        raise ValueError(f"unknown profile preset '{name}'; expected one of: {', '.join(PRESETS)}")
    return PRESETS[key]


@dataclass(frozen=True)
class Clearances:
    """Sizes derived from a profile. Every number is in metres unless named otherwise."""

    door_width: float           # clear width of a doorway: body plus a margin each side
    door_height: float          # clear height of a doorway: body plus headroom
    corridor_width: float       # two characters passing
    ceiling_height: float       # interior ceiling; third person also fits the camera
    stair_max_rise: float       # tallest riser
    stair_min_going: float      # shallowest tread
    takeoff_speed: float        # m/s, v0 = sqrt(2 g J)
    air_time: float             # s, 2 v0 / g
    jump_reach: float           # flat gap crossed at run speed
    sight_distance: float       # see a hazard in time to stop

    @classmethod
    def of(cls, p: PlayerProfile) -> "Clearances":
        v0 = math.sqrt(2.0 * p.gravity * p.jump_height)
        air = 2.0 * v0 / p.gravity
        cam = p.camera
        ceiling = p.height + HEAD_CLEARANCE
        if cam.mode == "third_person":
            # The camera sits behind and above the pivot; keep it under the ceiling.
            ceiling = max(ceiling, cam.height + cam.boom * math.sin(math.radians(abs(cam.pitch_deg))) + p.radius)
        stop = p.run_speed ** 2 / (2.0 * DEFAULT_BRAKE)
        return cls(
            door_width=round(2 * p.radius + 2 * SIDE_CLEARANCE, 4),
            door_height=round(p.height + HEAD_CLEARANCE, 4),
            corridor_width=round(4 * p.radius + 3 * SIDE_CLEARANCE, 4),
            ceiling_height=round(ceiling, 4),
            stair_max_rise=round(p.step_height, 4),
            stair_min_going=round(1.6 * p.radius, 4),
            takeoff_speed=round(v0, 4),
            air_time=round(air, 4),
            jump_reach=round(p.run_speed * air, 4),
            sight_distance=round(p.run_speed * p.reaction_s + stop, 4),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def resolve(value: Any) -> Optional[PlayerProfile]:
    """``None`` stays ``None``; a name, dict or profile becomes a ``PlayerProfile``."""
    if value is None:
        return None
    if isinstance(value, PlayerProfile):
        return value
    return PlayerProfile.from_dict(value)


__all__ = ["PROFILE_SCHEMA", "CAMERA_MODES", "Camera", "PlayerProfile", "Clearances", "PRESETS",
           "preset", "resolve"]
