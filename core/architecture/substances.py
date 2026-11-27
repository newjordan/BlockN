"""Building substances: physical materials with mass, strength and weathering.

A placement's ``material`` field is only an RGBA colour.  A building game or
simulation also needs a physical substance: how heavy a member is, how well it
resists crushing, how it bears load, and how its surface ages.  This module is
the single registry of those substances.

The numbers are honest typical values drawn from standard engineering-handbook
ranges, never invented precision.  A ``Substance`` records a typical density
with a low/high range; strengths and stiffness are single typical values that a
designer should treat as order-of-magnitude, not certified capacities.  The
``basis`` note on every substance names the family of sources the values come
from.

Geometry convention: the pipeline authors human-scale members in metres, so
densities are kg/m^3, strengths are MPa, elastic moduli are GPa, and one block
unit is treated as one metre for physical purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Substance:
    """One physical building material."""

    id: str
    name: str
    category: str
    # Mass
    density: float                       # kg/m3, typical
    density_range: Tuple[float, float]    # kg/m3, low and high
    # Strength and stiffness
    compressive_strength: float          # MPa, typical
    tensile_strength: float               # MPa, typical
    elastic_modulus: float                # GPa, typical
    # Contact behaviour
    friction: float                       # static friction coefficient
    # PBR appearance
    color_low: Tuple[int, int, int]       # sRGB, weathered/low end
    color_high: Tuple[int, int, int]      # sRGB, fresh/high end
    roughness: float
    metallic: float
    # How the surface ages, stated as data
    weathering: Tuple[str, ...]
    # Where the values come from
    basis: str


@dataclass(frozen=True)
class TintDrive:
    """One weathering tint and how hard age and exposure push toward it.

    ``amount = clamp(age * (age_drive + exposure_drive * channel), 0, 1)``
    where ``channel`` is the exposure selected for that tint (rain, shelter,
    or open sky) and both drives are 0..1. At age 0 the amount is 0.
    """

    color: Tuple[int, int, int]
    exposure: float
    age: float


@dataclass(frozen=True)
class WeatheringProfile:
    """Numeric weathering for one substance. The prose stays on ``Substance.weathering``."""

    substance_id: str
    biofilm: TintDrive          # green, on rain-washed upward faces
    soot: TintDrive             # black, on sheltered faces
    patina: TintDrive           # signature tint: verdigris, silver, lead white, or a local stain
    patina_on: str              # "open" (sun and rain) or "shelter"
    erosion_color: Tuple[int, int, int]
    roughness_age: float        # added to roughness at age 1
    grime_strength: float       # 0..1, band at the base
    grime_height: float         # metres above ground the band climbs
    erosion_strength: float     # 0..1, mixed by rain exposure and age


# Target tints shared by the profiles. Drives decide whether a substance uses them.
BIOFILM_GREEN = (78, 112, 68)
SOOT_BLACK = (36, 34, 32)
VERDIGRIS = (72, 150, 122)
SILVER_GREY = (174, 172, 164)
LEAD_WHITE = (228, 226, 216)
IRON_RED = (176, 92, 54)
OXIDE_BROWN = (122, 86, 58)
PALE_WASH = (214, 208, 196)
CORTEX_WHITE = (206, 204, 196)
RUST = (150, 78, 46)

# Applied finishes. The authored colour is the pigment or glaze, so a tone may
# change luminance and must not repaint the hue from the substance range.
PIGMENTED_SUBSTANCES = frozenset({
    "glazed-tile",
    "lime-plaster",
    "painted-linen",
    "painted-timber",
    "stucco",
    "window-glass",
})
# Half-width of the luminance scale across the eight tones: index 0 is -6%,
# index 7 is +6%, and the middle of the ramp is the authored colour.
TONE_LUMINANCE_SPAN = 0.06
# Natural substances place the authored colour on the segment from color_low
# to color_high and spread the eight tones across a window of this width.
NATURAL_TONE_WINDOW = 0.5
FINISH_TONE_COUNT = 8
# RGB distance at which an off-segment finish stops contributing hue. Buff and
# rose brick both project past the pale end of fired clay, and this reach keeps
# that hue apart. A stylised colour such as (0, 255, 65) is beyond the reach,
# so it contributes nothing and stays inside the real range.
FINISH_HUE_REACH = 160.0


def _tint(color: Tuple[int, int, int], exposure: float, age: float) -> TintDrive:
    return TintDrive(color=color, exposure=float(exposure), age=float(age))


def _profile(
    substance_id: str,
    biofilm: Tuple[float, float],
    soot: Tuple[float, float],
    patina_color: Tuple[int, int, int],
    patina: Tuple[float, float],
    patina_on: str,
    erosion_color: Tuple[int, int, int],
    erosion_strength: float,
    roughness_age: float,
    grime_strength: float,
    grime_height: float,
) -> WeatheringProfile:
    return WeatheringProfile(
        substance_id=substance_id,
        biofilm=_tint(BIOFILM_GREEN, *biofilm),
        soot=_tint(SOOT_BLACK, *soot),
        patina=_tint(patina_color, *patina),
        patina_on=patina_on,
        erosion_color=erosion_color,
        roughness_age=float(roughness_age),
        grime_strength=float(grime_strength),
        grime_height=float(grime_height),
        erosion_strength=float(erosion_strength),
    )


_CATEGORIES = ("stone", "earth", "soil", "binder", "timber", "roofing", "metal", "glass", "textile")


SUBSTANCES: Dict[str, Substance] = {substance.id: substance for substance in (
    # ------------------------------------------------------------- stone
    Substance(
        id="limestone", name="Limestone", category="stone",
        density=2200.0, density_range=(1700.0, 2600.0),
        compressive_strength=55.0, tensile_strength=3.5, elastic_modulus=30.0,
        friction=0.65,
        color_low=(196, 188, 168), color_high=(232, 226, 210),
        roughness=0.85, metallic=0.0,
        weathering=(
            "sheltered faces darken as soot and dust lodge in the pores",
            "biofilm and algae streak damp faces green within decades",
            "rainwashed faces stay pale while sheltered mouldings darken",
        ),
        basis="typical engineering-handbook range for sedimentary limestone",
    ),
    Substance(
        id="sandstone", name="Sandstone", category="stone",
        density=2200.0, density_range=(1900.0, 2500.0),
        compressive_strength=60.0, tensile_strength=3.0, elastic_modulus=20.0,
        friction=0.60,
        color_low=(188, 150, 108), color_high=(222, 190, 150),
        roughness=0.90, metallic=0.0,
        weathering=(
            "grain-by-grain erosion rounds carved detail",
            "iron cements redden and case-harden exposed faces",
            "salts crystallize under the surface and lift contour scales",
        ),
        basis="typical engineering-handbook range for quartz sandstone",
    ),
    Substance(
        id="granite", name="Granite", category="stone",
        density=2700.0, density_range=(2600.0, 2800.0),
        compressive_strength=160.0, tensile_strength=10.0, elastic_modulus=60.0,
        friction=0.70,
        color_low=(140, 135, 130), color_high=(185, 175, 168),
        roughness=0.75, metallic=0.0,
        weathering=(
            "feldspar weathers to kaolin and dulls the polish",
            "salt and frost open joints so outer slabs exfoliate",
            "quartz grains stand proud as the feldspar recesses",
        ),
        basis="typical engineering-handbook range for building granite",
    ),
    Substance(
        id="marble", name="Marble", category="stone",
        density=2700.0, density_range=(2600.0, 2800.0),
        compressive_strength=100.0, tensile_strength=7.0, elastic_modulus=55.0,
        friction=0.65,
        color_low=(210, 205, 198), color_high=(244, 242, 238),
        roughness=0.40, metallic=0.0,
        weathering=(
            "polish loses its sheen as calcite grains roughen",
            "acid rain etches and sugars the surface",
            "gypsum crusts blacken in polluted air",
        ),
        basis="typical engineering-handbook range for calcitic marble",
    ),
    Substance(
        id="basalt", name="Basalt", category="stone",
        density=2900.0, density_range=(2800.0, 3000.0),
        compressive_strength=150.0, tensile_strength=10.0, elastic_modulus=70.0,
        friction=0.70,
        color_low=(58, 58, 60), color_high=(84, 84, 88),
        roughness=0.85, metallic=0.0,
        weathering=(
            "iron oxides tint the weathering skin brown",
            "chemical decay turns the crust to clay, leaving a dull banded rind",
        ),
        basis="typical engineering-handbook range for dense basalt",
    ),
    Substance(
        id="tuff", name="Volcanic tuff", category="stone",
        density=1500.0, density_range=(1200.0, 1800.0),
        compressive_strength=10.0, tensile_strength=0.8, elastic_modulus=8.0,
        friction=0.60,
        color_low=(168, 156, 132), color_high=(198, 186, 162),
        roughness=0.95, metallic=0.0,
        weathering=(
            "soft ash grains rub away under rain and touch",
            "faces case-harden where silica cements the outer millimetres",
        ),
        basis="typical range for consolidated volcanic tuff used as lightweight ashlar",
    ),
    Substance(
        id="flint", name="Flint", category="stone",
        density=2600.0, density_range=(2500.0, 2700.0),
        compressive_strength=200.0, tensile_strength=15.0, elastic_modulus=70.0,
        friction=0.65,
        color_low=(54, 54, 58), color_high=(90, 88, 96),
        roughness=0.50, metallic=0.0,
        weathering=(
            "chalky cortex weathers white against the dark core",
            "fresh conchoidal faces stay glassy and blue-grey",
        ),
        basis="typical range for cryptocrystalline silica (flint nodules)",
    ),
    # ------------------------------------------------------ earth and fired
    Substance(
        id="fired-clay-brick", name="Fired clay brick", category="earth",
        density=1800.0, density_range=(1600.0, 2000.0),
        compressive_strength=30.0, tensile_strength=2.0, elastic_modulus=15.0,
        friction=0.70,
        color_low=(146, 74, 54), color_high=(186, 108, 80),
        roughness=0.90, metallic=0.0,
        weathering=(
            "frost and salt spall the base courses where water climbs",
            "soot darkens sheltered stretches",
            "lime run-off from the mortar streaks joints pale",
        ),
        basis="typical engineering-handbook range for common fired clay brick",
    ),
    Substance(
        id="engineering-brick", name="Engineering brick", category="earth",
        density=2000.0, density_range=(1900.0, 2200.0),
        compressive_strength=80.0, tensile_strength=4.0, elastic_modulus=25.0,
        friction=0.75,
        color_low=(98, 62, 52), color_high=(130, 84, 70),
        roughness=0.80, metallic=0.0,
        weathering=(
            "the hard-fired skin keeps its colour",
            "mortar joints erode faster than the brick itself",
        ),
        basis="typical range for Class A/Class B engineering brick",
    ),
    Substance(
        id="mud-brick", name="Mud brick (adobe)", category="earth",
        density=1550.0, density_range=(1300.0, 1800.0),
        compressive_strength=3.0, tensile_strength=0.3, elastic_modulus=1.5,
        friction=0.55,
        color_low=(168, 130, 92), color_high=(198, 158, 114),
        roughness=0.95, metallic=0.0,
        weathering=(
            "rain cuts channels at unrendered heads",
            "rising water and salt attack the base courses first",
            "a rendered skin protects while intact and sheds once cracked",
        ),
        basis="typical range for sun-dried mud brick and rammed earth",
    ),
    Substance(
        id="earth-fill", name="Compacted earth fill", category="soil",
        density=1800.0, density_range=(1500.0, 2100.0),
        compressive_strength=1.0, tensile_strength=0.02, elastic_modulus=0.05,
        friction=0.58,
        color_low=(88, 96, 64), color_high=(132, 120, 88),
        roughness=1.0, metallic=0.0,
        weathering=(
            "turf binds the surface and greens the slopes within a season",
            "rain gullies the steepest faces where the turf breaks",
            "the crest slumps and spreads slowly over the centuries",
        ),
        basis="typical range for dense compacted soil and rubble fill; strength is the ultimate bearing capacity under a wide base, checked with a geotechnical factor of safety",
    ),
    Substance(
        id="terracotta", name="Terracotta", category="earth",
        density=1900.0, density_range=(1800.0, 2000.0),
        compressive_strength=30.0, tensile_strength=3.0, elastic_modulus=15.0,
        friction=0.65,
        color_low=(172, 96, 62), color_high=(204, 130, 92),
        roughness=0.70, metallic=0.0,
        weathering=(
            "the fired skin crazes and pits over decades",
            "rusting steel fixings stain the ceramic",
        ),
        basis="typical range for fired architectural terracotta",
    ),
    Substance(
        id="glazed-tile", name="Glazed tile", category="earth",
        density=2000.0, density_range=(1900.0, 2100.0),
        compressive_strength=40.0, tensile_strength=4.0, elastic_modulus=25.0,
        friction=0.40,
        color_low=(86, 142, 148), color_high=(222, 236, 232),
        roughness=0.15, metallic=0.0,
        weathering=(
            "glaze pits and loses gloss where rain stands",
            "crazing lets dirt ink the crack network",
        ),
        basis="typical range for lead- or tin-glazed ceramic tile",
    ),
    # ------------------------------------------------------------ binders
    Substance(
        id="lime-mortar", name="Lime mortar", category="binder",
        density=1650.0, density_range=(1500.0, 1800.0),
        compressive_strength=2.0, tensile_strength=0.2, elastic_modulus=2.0,
        friction=0.60,
        color_low=(196, 190, 172), color_high=(228, 224, 210),
        roughness=0.90, metallic=0.0,
        weathering=(
            "carbonates slowly and stays comparatively soft",
            "washes out of deep joints, leaving the stone proud",
        ),
        basis="typical range for carbonated lime mortar (NHL 2 class)",
    ),
    Substance(
        id="lime-plaster", name="Lime plaster", category="binder",
        density=1600.0, density_range=(1500.0, 1750.0),
        compressive_strength=2.5, tensile_strength=0.5, elastic_modulus=4.0,
        friction=0.60,
        color_low=(222, 216, 202), color_high=(240, 238, 230),
        roughness=0.85, metallic=0.0,
        weathering=(
            "limewash coats renew the pale face",
            "damp climbs and blooms salts at the base",
        ),
        basis="typical range for lime plaster and limewash",
    ),
    Substance(
        id="stucco", name="Stucco", category="binder",
        density=1850.0, density_range=(1700.0, 2000.0),
        compressive_strength=5.0, tensile_strength=1.0, elastic_modulus=8.0,
        friction=0.60,
        color_low=(208, 198, 178), color_high=(232, 226, 212),
        roughness=0.85, metallic=0.0,
        weathering=(
            "hairline crazing maps the frame behind",
            "corner beads and reveals chip first",
        ),
        basis="typical range for cement/lime stucco render",
    ),
    Substance(
        id="roman-concrete", name="Roman concrete (opus caementicium)", category="binder",
        density=1700.0, density_range=(1500.0, 1900.0),
        compressive_strength=8.0, tensile_strength=0.7, elastic_modulus=10.0,
        friction=0.65,
        color_low=(168, 156, 138), color_high=(196, 186, 170),
        roughness=0.95, metallic=0.0,
        weathering=(
            "the pozzolanic skin hardens further under water",
            "brick facings spall where the core carbonates unevenly",
            "rendered-off faces expose the tuff aggregate",
        ),
        basis="typical range for opus caementicium with tuff and pozzolana aggregate",
    ),
    # ------------------------------------------------------------ timber
    Substance(
        id="oak", name="Oak", category="timber",
        density=750.0, density_range=(680.0, 820.0),
        compressive_strength=50.0, tensile_strength=90.0, elastic_modulus=12.0,
        friction=0.45,
        color_low=(110, 82, 54), color_high=(158, 120, 78),
        roughness=0.60, metallic=0.0,
        weathering=(
            "silvers to grey as sunlight degrades the lignin",
            "checks open along the rays as it seasons in place",
            "wet-dry cycles rot the sapwood first",
        ),
        basis="typical range for structural oak, strengths parallel to grain",
    ),
    Substance(
        id="softwood", name="Softwood (pine/fir)", category="timber",
        density=500.0, density_range=(420.0, 580.0),
        compressive_strength=35.0, tensile_strength=70.0, elastic_modulus=11.0,
        friction=0.40,
        color_low=(168, 132, 90), color_high=(204, 172, 128),
        roughness=0.65, metallic=0.0,
        weathering=(
            "faces silver blue-grey within a year of sun",
            "resin bleeds at knots in heat",
            "damp sapwood falls to beetle and rot",
        ),
        basis="typical range for structural pine and fir, strengths parallel to grain",
    ),
    Substance(
        id="cedar", name="Cedar / hinoki", category="timber",
        density=420.0, density_range=(350.0, 500.0),
        compressive_strength=30.0, tensile_strength=60.0, elastic_modulus=8.5,
        friction=0.40,
        color_low=(192, 164, 122), color_high=(224, 200, 158),
        roughness=0.60, metallic=0.0,
        weathering=(
            "aromatic heartwood resists rot for centuries",
            "colour mellows from pale straw to warm amber",
            "unprotected faces silver like any other timber",
        ),
        basis="typical range for cedar and hinoki heartwood, strengths parallel to grain",
    ),
    Substance(
        id="painted-timber", name="Painted timber", category="timber",
        density=520.0, density_range=(420.0, 750.0),
        compressive_strength=35.0, tensile_strength=70.0, elastic_modulus=11.0,
        friction=0.45,
        color_low=(128, 40, 34), color_high=(214, 72, 58),
        roughness=0.40, metallic=0.0,
        weathering=(
            "lacquer and mineral paint chalk and fade in sun",
            "paint flakes at edges and joints and bares silvered timber",
            "sheltered colour keeps its depth and slowly darkens",
        ),
        basis="typical range for softwood under lacquer or mineral paint; strength is that of the fir beneath the coating",
    ),
    # ----------------------------------------------------------- roofing
    Substance(
        id="slate", name="Roofing slate", category="roofing",
        density=2700.0, density_range=(2600.0, 2800.0),
        compressive_strength=120.0, tensile_strength=15.0, elastic_modulus=20.0,
        friction=0.55,
        color_low=(58, 62, 70), color_high=(94, 100, 110),
        roughness=0.60, metallic=0.0,
        weathering=(
            "pyrite crystals rust to golden spots",
            "flakes delaminate and drop from weathered laps",
            "blue-grey deepens where rain washes fastest",
        ),
        basis="typical range for roofing slate",
    ),
    Substance(
        id="clay-roof-tile", name="Clay roof tile", category="roofing",
        density=1900.0, density_range=(1800.0, 2000.0),
        compressive_strength=25.0, tensile_strength=3.0, elastic_modulus=15.0,
        friction=0.65,
        color_low=(136, 76, 56), color_high=(188, 110, 74),
        roughness=0.85, metallic=0.0,
        weathering=(
            "moss and lichen colonize shaded laps",
            "frost cracks tiles that hold water",
            "colour darkens where consistently damp",
        ),
        basis="typical range for fired clay roof tile",
    ),
    Substance(
        id="thatch", name="Thatch", category="roofing",
        density=150.0, density_range=(90.0, 220.0),
        compressive_strength=0.8, tensile_strength=2.0, elastic_modulus=0.15,
        friction=0.55,
        color_low=(150, 118, 62), color_high=(196, 164, 96),
        roughness=0.95, metallic=0.0,
        weathering=(
            "straw darkens from gold to grey-brown within seasons",
            "the ridge wears first and needs re-combing",
            "birds pull straw where netting frays",
        ),
        basis="bulk values for bundled straw and reed thatch; a covering, not a structure",
    ),
    # ------------------------------------------------------------ metals
    Substance(
        id="lead-sheet", name="Lead sheet", category="metal",
        density=11340.0, density_range=(11300.0, 11400.0),
        compressive_strength=12.0, tensile_strength=15.0, elastic_modulus=16.0,
        friction=0.45,
        color_low=(108, 110, 116), color_high=(140, 142, 148),
        roughness=0.50, metallic=0.90,
        weathering=(
            "a bright surface whitens to basic lead carbonate within weeks",
            "sheet creeps and wrinkles under its own weight",
            "run-off streaks the stone below grey",
        ),
        basis="typical values for rolled lead sheet, yield-controlled",
    ),
    Substance(
        id="copper-sheet", name="Copper sheet", category="metal",
        density=8960.0, density_range=(8900.0, 9000.0),
        compressive_strength=70.0, tensile_strength=220.0, elastic_modulus=120.0,
        friction=0.40,
        color_low=(94, 62, 48), color_high=(184, 115, 90),
        roughness=0.35, metallic=1.0,
        weathering=(
            "the bright surface dulls to oxide within weeks",
            "a brown patina forms within a few years",
            "sheltered faces green to verdigris over decades",
        ),
        basis="typical values for annealed copper sheet",
    ),
    Substance(
        id="wrought-iron", name="Wrought iron", category="metal",
        density=7800.0, density_range=(7700.0, 7900.0),
        compressive_strength=200.0, tensile_strength=350.0, elastic_modulus=190.0,
        friction=0.50,
        color_low=(76, 74, 72), color_high=(108, 104, 100),
        roughness=0.50, metallic=1.0,
        weathering=(
            "mill scale flakes to orange rust without paint",
            "welded joints weather first",
            "paint chips at edges and bolt heads",
        ),
        basis="typical values for wrought iron",
    ),
    # ------------------------------------------------------------- glass
    Substance(
        id="window-glass", name="Window glass", category="glass",
        density=2500.0, density_range=(2400.0, 2600.0),
        compressive_strength=800.0, tensile_strength=45.0, elastic_modulus=70.0,
        friction=0.35,
        color_low=(150, 190, 200), color_high=(200, 228, 236),
        roughness=0.05, metallic=0.0,
        weathering=(
            "alkali leaches from old panes, pitting and iridescing them",
            "leadlight corrosion clouds the pane edges",
        ),
        basis="typical values for soda-lime window glass; design tensile stress is far lower",
    ),
    # ----------------------------------------------------------- textile
    Substance(
        id="painted-linen", name="Painted linen", category="textile",
        density=420.0, density_range=(280.0, 600.0),
        compressive_strength=1.0, tensile_strength=40.0, elastic_modulus=1.5,
        friction=0.35,
        color_low=(168, 52, 56), color_high=(204, 84, 88),
        roughness=0.90, metallic=0.0,
        weathering=(
            "pigment fades unevenly in the sun",
            "hems fray and tear at the mast in wind",
        ),
        basis="representative values for painted woven flax; banners are non-structural",
    ),
)}


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _clamp_byte(value: float) -> int:
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return int(round(value))


def finish_tone(
    substance: Substance,
    authored: Tuple[int, int, int],
    index: int,
) -> Tuple[int, int, int]:
    """sRGB tone for one member, from the authored finish and the substance.

    Pigmented and glazed substances keep the authored hue and vary only
    luminance, by ``TONE_LUMINANCE_SPAN`` across the eight tones. Natural
    substances project the authored RGB onto ``color_low`` → ``color_high``,
    centre a window of ``NATURAL_TONE_WINDOW`` on that position, and spread the
    tones across the window. Hue that the segment cannot express is kept while
    the authored colour is within ``FINISH_HUE_REACH``, and dropped beyond it.
    """
    red, green, blue = float(authored[0]), float(authored[1]), float(authored[2])
    blend = (int(index) % FINISH_TONE_COUNT) / float(FINISH_TONE_COUNT - 1)
    if substance.id in PIGMENTED_SUBSTANCES:
        scale = 1.0 - TONE_LUMINANCE_SPAN + 2.0 * TONE_LUMINANCE_SPAN * blend
        return (
            _clamp_byte(red * scale),
            _clamp_byte(green * scale),
            _clamp_byte(blue * scale),
        )

    low = substance.color_low
    high = substance.color_high
    delta = (float(high[0] - low[0]), float(high[1] - low[1]), float(high[2] - low[2]))
    offset = (red - float(low[0]), green - float(low[1]), blue - float(low[2]))
    denom = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
    if denom <= 1e-12:
        position = 0.0
        projected = (float(low[0]), float(low[1]), float(low[2]))
    else:
        position = _clamp01((offset[0] * delta[0] + offset[1] * delta[1] + offset[2] * delta[2]) / denom)
        projected = (
            float(low[0]) + delta[0] * position,
            float(low[1]) + delta[1] * position,
            float(low[2]) + delta[2] * position,
        )
    half = NATURAL_TONE_WINDOW * 0.5
    start = _clamp01(position - half)
    if start > 1.0 - NATURAL_TONE_WINDOW:
        start = 1.0 - NATURAL_TONE_WINDOW
    place = start + NATURAL_TONE_WINDOW * blend
    on_segment = (
        float(low[0]) + delta[0] * place,
        float(low[1]) + delta[1] * place,
        float(low[2]) + delta[2] * place,
    )
    residual = (red - projected[0], green - projected[1], blue - projected[2])
    distance = (residual[0] ** 2 + residual[1] ** 2 + residual[2] ** 2) ** 0.5
    keep = 0.0 if FINISH_HUE_REACH <= 0.0 else _clamp01(1.0 - distance / FINISH_HUE_REACH)
    return tuple(
        _clamp_byte(on_segment[channel] + residual[channel] * keep) for channel in range(3)
    )


# Drives are (exposure, age). Their sum stays at most 1 so a fully exposed,
# fully aged face reaches the tint without wrapping past it. Biofilm and soot
# age drives stay at or below 0.08 so a face with no exposure only drifts.
# Lead whitening and copper verdigris keep a larger age drive: that patina is
# the material, and exposure still decides which faces take it.
WEATHERING = {profile.substance_id: profile for profile in (
    _profile("limestone", (0.68, 0.08), (0.42, 0.08), PALE_WASH, (0.06, 0.04), "open",
             OXIDE_BROWN, 0.08, 0.08, 0.55, 1.0),
    _profile("sandstone", (0.22, 0.08), (0.16, 0.08), IRON_RED, (0.08, 0.04), "open",
             IRON_RED, 0.78, 0.12, 0.42, 1.0),
    _profile("granite", (0.18, 0.08), (0.22, 0.08), OXIDE_BROWN, (0.06, 0.04), "open",
             OXIDE_BROWN, 0.16, 0.10, 0.28, 0.8),
    _profile("marble", (0.12, 0.06), (0.40, 0.08), SOOT_BLACK, (0.16, 0.10), "shelter",
             PALE_WASH, 0.14, 0.36, 0.32, 0.8),
    _profile("basalt", (0.10, 0.04), (0.14, 0.08), OXIDE_BROWN, (0.22, 0.10), "open",
             OXIDE_BROWN, 0.36, 0.06, 0.22, 0.6),
    _profile("tuff", (0.28, 0.08), (0.18, 0.08), PALE_WASH, (0.10, 0.06), "open",
             OXIDE_BROWN, 0.48, 0.08, 0.38, 1.0),
    _profile("flint", (0.06, 0.02), (0.10, 0.06), CORTEX_WHITE, (0.36, 0.12), "open",
             PALE_WASH, 0.06, 0.04, 0.16, 0.4),
    _profile("fired-clay-brick", (0.28, 0.08), (0.40, 0.08), PALE_WASH, (0.14, 0.08), "open",
             IRON_RED, 0.16, 0.06, 0.66, 1.0),
    _profile("engineering-brick", (0.08, 0.04), (0.12, 0.08), OXIDE_BROWN, (0.04, 0.02), "open",
             OXIDE_BROWN, 0.05, 0.03, 0.24, 0.6),
    _profile("mud-brick", (0.20, 0.08), (0.18, 0.08), OXIDE_BROWN, (0.10, 0.06), "open",
             OXIDE_BROWN, 0.72, 0.10, 0.82, 1.2),
    _profile("earth-fill", (0.70, 0.08), (0.10, 0.06), BIOFILM_GREEN, (0.46, 0.12), "open",
             OXIDE_BROWN, 0.22, 0.02, 0.18, 0.4),
    _profile("terracotta", (0.18, 0.08), (0.22, 0.08), RUST, (0.16, 0.08), "open",
             RUST, 0.20, 0.10, 0.30, 0.8),
    _profile("glazed-tile", (0.12, 0.04), (0.24, 0.08), SOOT_BLACK, (0.10, 0.08), "open",
             OXIDE_BROWN, 0.05, 0.22, 0.14, 0.3),
    _profile("lime-mortar", (0.32, 0.08), (0.28, 0.08), PALE_WASH, (0.08, 0.04), "open",
             PALE_WASH, 0.10, 0.08, 0.48, 1.0),
    _profile("lime-plaster", (0.24, 0.08), (0.20, 0.08), PALE_WASH, (0.12, 0.08), "open",
             PALE_WASH, 0.08, 0.06, 0.52, 1.0),
    _profile("stucco", (0.18, 0.08), (0.28, 0.08), PALE_WASH, (0.08, 0.04), "open",
             OXIDE_BROWN, 0.12, 0.10, 0.40, 0.8),
    _profile("roman-concrete", (0.18, 0.08), (0.22, 0.08), OXIDE_BROWN, (0.14, 0.08), "open",
             OXIDE_BROWN, 0.20, 0.05, 0.34, 0.8),
    _profile("oak", (0.10, 0.04), (0.18, 0.08), SILVER_GREY, (0.62, 0.12), "open",
             OXIDE_BROWN, 0.08, 0.18, 0.30, 0.6),
    _profile("softwood", (0.14, 0.06), (0.14, 0.08), SILVER_GREY, (0.58, 0.12), "open",
             OXIDE_BROWN, 0.06, 0.16, 0.26, 0.5),
    _profile("cedar", (0.08, 0.04), (0.12, 0.06), SILVER_GREY, (0.52, 0.12), "open",
             (176, 140, 88), 0.05, 0.12, 0.20, 0.5),
    # A coating ages by failing: exposed paint silvers where it flakes, so the
    # patina is the timber showing through and roughness climbs as it chalks.
    _profile("painted-timber", (0.10, 0.04), (0.16, 0.08), SILVER_GREY, (0.40, 0.10), "open",
             (168, 150, 132), 0.08, 0.34, 0.22, 0.6),
    _profile("slate", (0.18, 0.06), (0.14, 0.08), (70, 78, 88), (0.22, 0.10), "open",
             RUST, 0.10, 0.08, 0.18, 0.3),
    _profile("clay-roof-tile", (0.52, 0.08), (0.18, 0.08), BIOFILM_GREEN, (0.28, 0.10), "open",
             IRON_RED, 0.12, 0.08, 0.22, 0.3),
    _profile("thatch", (0.22, 0.08), (0.32, 0.08), (132, 118, 90), (0.40, 0.12), "open",
             (132, 118, 90), 0.14, 0.04, 0.14, 0.2),
    _profile("lead-sheet", (0.02, 0.02), (0.08, 0.04), LEAD_WHITE, (0.22, 0.70), "open",
             PALE_WASH, 0.02, 0.18, 0.10, 0.2),
    _profile("copper-sheet", (0.04, 0.02), (0.08, 0.04), VERDIGRIS, (0.40, 0.52), "shelter",
             OXIDE_BROWN, 0.05, 0.24, 0.08, 0.2),
    _profile("wrought-iron", (0.02, 0.02), (0.14, 0.08), RUST, (0.42, 0.12), "open",
             RUST, 0.36, 0.28, 0.20, 0.4),
    _profile("window-glass", (0.06, 0.02), (0.10, 0.08), PALE_WASH, (0.14, 0.10), "open",
             PALE_WASH, 0.04, 0.12, 0.08, 0.2),
    _profile("painted-linen", (0.12, 0.04), (0.16, 0.08), (188, 150, 140), (0.36, 0.12), "open",
             (188, 150, 140), 0.10, 0.05, 0.10, 0.2),
)}


def _validate_weathering() -> None:
    missing = sorted(set(SUBSTANCES) - set(WEATHERING))
    extra = sorted(set(WEATHERING) - set(SUBSTANCES))
    if missing or extra:
        raise RuntimeError(
            f"weathering profiles do not match the substance registry: missing {missing}, extra {extra}"
        )
    unknown_pigment = sorted(PIGMENTED_SUBSTANCES - set(SUBSTANCES))
    if unknown_pigment:
        raise RuntimeError(f"pigmented substances are not in the registry: {unknown_pigment}")
    for profile in WEATHERING.values():
        if profile.patina_on not in {"open", "shelter"}:
            raise RuntimeError(f"{profile.substance_id} patina_on must be 'open' or 'shelter'")
        if profile.grime_height <= 0.0:
            raise RuntimeError(f"{profile.substance_id} grime_height must be positive")
        for name, value in (
            ("grime_strength", profile.grime_strength),
            ("erosion_strength", profile.erosion_strength),
        ):
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{profile.substance_id} {name} must be in 0..1")
        if not -0.2 <= profile.roughness_age <= 1.0:
            raise RuntimeError(f"{profile.substance_id} roughness_age is out of range")
        for label, tint in (("biofilm", profile.biofilm), ("soot", profile.soot), ("patina", profile.patina)):
            if not 0.0 <= tint.exposure <= 1.0 or not 0.0 <= tint.age <= 1.0:
                raise RuntimeError(f"{profile.substance_id} {label} drives must be in 0..1")
            if tint.exposure + tint.age > 1.0 + 1e-9:
                raise RuntimeError(f"{profile.substance_id} {label} drives sum past 1")
        if profile.biofilm.age > 0.08 + 1e-9:
            raise RuntimeError(f"{profile.substance_id} biofilm age drive must be at most 0.08")
        if profile.soot.age > 0.08 + 1e-9:
            raise RuntimeError(f"{profile.substance_id} soot age drive must be at most 0.08")


_validate_weathering()


REQUIRED_SUBSTANCE_IDS = (
    # stone
    "limestone", "sandstone", "granite", "marble", "basalt", "tuff", "flint",
    # earth and fired
    "fired-clay-brick", "engineering-brick", "mud-brick", "terracotta", "glazed-tile",
    # binders
    "lime-mortar", "lime-plaster", "stucco", "roman-concrete",
    # timber
    "oak", "softwood", "cedar",
    # roofing
    "slate", "clay-roof-tile", "thatch",
    # metals
    "lead-sheet", "copper-sheet", "wrought-iron",
    # glass
    "window-glass",
)


def get_substance(substance_id: str) -> Substance:
    """Return one substance by id; raise ``KeyError`` for an unknown id."""
    return SUBSTANCES[str(substance_id)]


def substance_exists(substance_id: str) -> bool:
    return str(substance_id) in SUBSTANCES


__all__ = [
    "Substance",
    "TintDrive",
    "WeatheringProfile",
    "SUBSTANCES",
    "WEATHERING",
    "BIOFILM_GREEN",
    "SOOT_BLACK",
    "VERDIGRIS",
    "SILVER_GREY",
    "LEAD_WHITE",
    "REQUIRED_SUBSTANCE_IDS",
    "PIGMENTED_SUBSTANCES",
    "TONE_LUMINANCE_SPAN",
    "NATURAL_TONE_WINDOW",
    "FINISH_HUE_REACH",
    "FINISH_TONE_COUNT",
    "finish_tone",
    "get_substance",
    "substance_exists",
]
