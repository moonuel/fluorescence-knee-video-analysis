"""
Centralized metadata for all knee videos: condition, cycles, and anatomical regions.

This module replaces scattered CYCLES, TYPES, and REGION_RANGES dictionaries.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

Phase = Literal["flexion", "extension", "both"]

@dataclass(frozen=True)
class FrameRange:
    """0-based, inclusive frame range in video array indices."""
    s: int  # inclusive
    e: int    # inclusive

    def to_slice(self) -> slice:
        """For numpy slicing: video[frame_range.to_slice()]."""
        return slice(self.s, self.e + 1)

    @classmethod
    def from_1based(cls, start_1: int, end_1: int) -> "FrameRange":
        """Construct from 1-based inclusive frame numbers (e.g. Excel)."""
        return cls(s=start_1 - 1, e=end_1 - 1)


@dataclass(frozen=True)
class Cycle:
    """One flexion–extension cycle in absolute frame indices."""
    flex: FrameRange
    ext: FrameRange

    def full_cycle(self) -> FrameRange:
        """Overall range flex-start → ext-end."""
        return FrameRange(s=self.flex.s, e=self.ext.e)


@dataclass(frozen=True)
class RegionSegments:
    """Segment index range for anatomical regions.

    These stay 1-based to match notes/Excel; we expose a helper
    to convert to 0-based slice for arrays shaped (N_segments, ...).
    """
    s: int  # 1-based inclusive
    e: int    # 1-based inclusive

    def to_index_slice(self) -> slice:
        """0-based slice: total_sums[region_slice, :]."""
        return slice(self.s - 1, self.e)


@dataclass(frozen=True)
class KneeVideoMeta:
    """All metadata for a single knee video at a given N."""
    condition: str                 # e.g. "normal", "aging", "dmm-0w", "dmm-4w"
    video_id: int                  # e.g. 308, 1339, 1195
    n_segments: int                # e.g. 64

    # Ordered list of flexion–extension cycles
    cycles: List[Cycle]

    # Anatomical regions: JC, OT, SB
    regions: Dict[str, RegionSegments]  # keys: "JC", "OT", "SB"

    # (Optional) extra per-video metadata
    # metadata: Dict[str, Any] = field(default_factory=dict)

    def get_cycle(self, idx: int) -> Cycle:
        return self.cycles[idx]

    def get_region(self, name: str) -> RegionSegments:
        return self.regions[name]


# Global registry: key = (condition, video_id, n_segments)
Key = Tuple[str, int, int]  # (condition, video_id, n_segments)

KNEE_VIDEOS: Dict[Key, KneeVideoMeta] = {
    # Normal knees, N=64
    ("normal", 308, 64): KneeVideoMeta(
        condition="normal",
        video_id=308,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(71, 116), FrameRange.from_1based(117, 155)),
            Cycle(FrameRange.from_1based(253, 298), FrameRange.from_1based(299, 335)),
            Cycle(FrameRange.from_1based(585, 618), FrameRange.from_1based(630, 669)),
            Cycle(FrameRange.from_1based(156, 199), FrameRange.from_1based(210, 250)),
            Cycle(FrameRange.from_1based(342, 393), FrameRange.from_1based(407, 439)),
        ],
        regions={
            "JC": RegionSegments(1, 29),
            "OT": RegionSegments(30, 47),
            "SB": RegionSegments(48, 64),
        },
    ),

    ("normal", 1190, 64): KneeVideoMeta(
        condition="normal",
        video_id=1190,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(66, 89), FrameRange.from_1based(92, 109)),
            Cycle(FrameRange.from_1based(421, 452), FrameRange.from_1based(470, 492)),
            Cycle(FrameRange.from_1based(503, 532), FrameRange.from_1based(533, 569)),
            Cycle(FrameRange.from_1based(737, 767), FrameRange.from_1based(770, 793)),
            Cycle(FrameRange.from_1based(794, 822), FrameRange.from_1based(823, 860)),
        ],
        regions={
            "JC": RegionSegments(1, 29),  # Assuming default; adjust if needed
            "OT": RegionSegments(30, 47),
            "SB": RegionSegments(48, 64),
        },
    ),

    ("normal", 1193, 64): KneeVideoMeta(
        condition="normal",
        video_id=1193,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(1792, 1801), FrameRange.from_1based(1802, 1812)),
            Cycle(FrameRange.from_1based(1813, 1822), FrameRange.from_1based(1823, 1833)),
            Cycle(FrameRange.from_1based(1834, 1843), FrameRange.from_1based(1844, 1852)),
            Cycle(FrameRange.from_1based(1853, 1863), FrameRange.from_1based(1864, 1872)),
            Cycle(FrameRange.from_1based(1873, 1881), FrameRange.from_1based(1881, 1889)),
        ],
        regions={
            "JC": RegionSegments(1, 29),  # Assuming default
            "OT": RegionSegments(30, 47),
            "SB": RegionSegments(48, 64),
        },
    ),

    ("normal", 1207, 64): KneeVideoMeta(
        condition="normal",
        video_id=1207,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(242, 254), FrameRange.from_1based(264, 280)),
            Cycle(FrameRange.from_1based(281, 293), FrameRange.from_1based(299, 312)),
            Cycle(FrameRange.from_1based(318, 335), FrameRange.from_1based(337, 352)),
            Cycle(FrameRange.from_1based(353, 372), FrameRange.from_1based(373, 389)),
            Cycle(FrameRange.from_1based(391, 411), FrameRange.from_1based(412, 431)),
            Cycle(FrameRange.from_1based(434, 451), FrameRange.from_1based(453, 467)),
            Cycle(FrameRange.from_1based(472, 486), FrameRange.from_1based(488, 505)),
            Cycle(FrameRange.from_1based(614, 632), FrameRange.from_1based(633, 651)),
            Cycle(FrameRange.from_1based(652, 671), FrameRange.from_1based(672, 690)),
            Cycle(FrameRange.from_1based(693, 708), FrameRange.from_1based(709, 727)),
            Cycle(FrameRange.from_1based(731, 748), FrameRange.from_1based(751, 767)),
            Cycle(FrameRange.from_1based(768, 786), FrameRange.from_1based(787, 804)),
            Cycle(FrameRange.from_1based(807, 822), FrameRange.from_1based(824, 841)),
            Cycle(FrameRange.from_1based(844, 862), FrameRange.from_1based(863, 877)),
        ],
        regions={
            "JC": RegionSegments(1, 26),
            "OT": RegionSegments(27, 39),
            "SB": RegionSegments(40, 64),
        },
    ),

    # Normal knees, N=16 (cycles = N=64; JC/OT/SB downscaled from 64-seg ranges)
    ("normal", 308, 16): KneeVideoMeta(
        condition="normal",
        video_id=308,
        n_segments=16,
        cycles=[  # same frames as ("normal", 308, 64)
            Cycle(FrameRange.from_1based(71, 116), FrameRange.from_1based(117, 155)),
            Cycle(FrameRange.from_1based(253, 298), FrameRange.from_1based(299, 335)),
            Cycle(FrameRange.from_1based(585, 618), FrameRange.from_1based(630, 669)),
            Cycle(FrameRange.from_1based(156, 199), FrameRange.from_1based(210, 250)),
        ],
        regions={
            "JC": RegionSegments(1, 7),
            "OT": RegionSegments(8, 12),
            "SB": RegionSegments(13, 16),
        },
    ),

    ("normal", 1190, 16): KneeVideoMeta(
        condition="normal",
        video_id=1190,
        n_segments=16,
        cycles=[  # same frames as ("normal", 1190, 64)
            Cycle(FrameRange.from_1based(66, 89), FrameRange.from_1based(92, 109)),
            Cycle(FrameRange.from_1based(421, 452), FrameRange.from_1based(470, 492)),
            Cycle(FrameRange.from_1based(503, 532), FrameRange.from_1based(533, 569)),
            Cycle(FrameRange.from_1based(737, 767), FrameRange.from_1based(770, 793)),
            Cycle(FrameRange.from_1based(794, 822), FrameRange.from_1based(823, 860)),
        ],
        regions={
            "JC": RegionSegments(1, 7),
            "OT": RegionSegments(8, 12),
            "SB": RegionSegments(13, 16),
        },
    ),

    ("normal", 1193, 16): KneeVideoMeta(
        condition="normal",
        video_id=1193,
        n_segments=16,
        cycles=[  # same frames as ("normal", 1193, 64)
            Cycle(FrameRange.from_1based(1792, 1801), FrameRange.from_1based(1802, 1812)),
            Cycle(FrameRange.from_1based(1813, 1822), FrameRange.from_1based(1823, 1833)),
            Cycle(FrameRange.from_1based(1834, 1843), FrameRange.from_1based(1844, 1852)),
            Cycle(FrameRange.from_1based(1853, 1863), FrameRange.from_1based(1864, 1872)),
            Cycle(FrameRange.from_1based(1873, 1881), FrameRange.from_1based(1881, 1889)),
        ],
        regions={
            "JC": RegionSegments(1, 7),
            "OT": RegionSegments(8, 12),
            "SB": RegionSegments(13, 16),
        },
    ),

    # Aging knees, N=64
    ("aging", 1339, 64): KneeVideoMeta(
        condition="aging",
        video_id=1339,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(290, 309), FrameRange.from_1based(312, 329)),
            Cycle(FrameRange.from_1based(331, 352), FrameRange.from_1based(355, 374)),
            Cycle(FrameRange.from_1based(375, 394), FrameRange.from_1based(398, 421)),
            Cycle(FrameRange.from_1based(422, 439), FrameRange.from_1based(441, 463)),
            Cycle(FrameRange.from_1based(464, 488), FrameRange.from_1based(490, 512)),
            Cycle(FrameRange.from_1based(513, 530), FrameRange.from_1based(532, 553)),
            Cycle(FrameRange.from_1based(554, 576), FrameRange.from_1based(579, 609)),
            # Revised cycles suggested by Huizhu on 2026-01-19
            Cycle(FrameRange.from_1based(1199, 1231), FrameRange.from_1based(1232, 1264)), # 8 
            Cycle(FrameRange.from_1based(1265, 1299), FrameRange.from_1based(1300, 1336)), # 9
            Cycle(FrameRange.from_1based(1337, 1365), FrameRange.from_1based(1366, 1390)), # 10
        ],
        regions={
            "JC": RegionSegments(1, 27),
            "OT": RegionSegments(28, 44),
            "SB": RegionSegments(45, 64),
        },
    ),

    ("aging", 1342, 64): KneeVideoMeta(
        condition="aging",
        video_id=1342,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(62, 81), FrameRange.from_1based(82, 100)),
            Cycle(FrameRange.from_1based(102, 119), FrameRange.from_1based(123, 151)),
            Cycle(FrameRange.from_1based(152, 171), FrameRange.from_1based(178, 199)),
            Cycle(FrameRange.from_1based(206, 222), FrameRange.from_1based(223, 246)),
            Cycle(FrameRange.from_1based(247, 272), FrameRange.from_1based(273, 297)),
            Cycle(FrameRange.from_1based(298, 320), FrameRange.from_1based(321, 340)),
            Cycle(FrameRange.from_1based(341, 364), FrameRange.from_1based(365, 384)),
        ],
        regions={
            "JC": RegionSegments(1, 27),  # Assuming default
            "OT": RegionSegments(28, 44),
            "SB": RegionSegments(45, 64),
        },
    ),

    ("aging", 1357, 64): KneeVideoMeta(
        condition="aging",
        video_id=1357,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(218, 240), FrameRange.from_1based(241, 272)),
            Cycle(FrameRange.from_1based(278, 305), FrameRange.from_1based(306, 330)),
            Cycle(FrameRange.from_1based(420, 447), FrameRange.from_1based(449, 467)),
            Cycle(FrameRange.from_1based(469, 492), FrameRange.from_1based(493, 517)),
            Cycle(FrameRange.from_1based(639, 660), FrameRange.from_1based(662, 682)),
            Cycle(FrameRange.from_1based(683, 709), FrameRange.from_1based(710, 732)),
            Cycle(FrameRange.from_1based(744, 775), FrameRange.from_1based(777, 779)),
            Cycle(FrameRange.from_1based(801, 828), FrameRange.from_1based(837, 858)),
            Cycle(FrameRange.from_1based(859, 890), FrameRange.from_1based(893, 917)),
            Cycle(FrameRange.from_1based(1067, 1091), FrameRange.from_1based(1092, 1118)),
            Cycle(FrameRange.from_1based(1136, 1171), FrameRange.from_1based(1173, 1198)),
            Cycle(FrameRange.from_1based(1199, 1230), FrameRange.from_1based(1232, 1260)),
            Cycle(FrameRange.from_1based(1261, 1285), FrameRange.from_1based(1286, 1311)),
            Cycle(FrameRange.from_1based(1313, 1340), FrameRange.from_1based(1342, 1365)),
            Cycle(FrameRange.from_1based(1368, 1394), FrameRange.from_1based(1395, 1419)),
        ],
        regions={
            "JC": RegionSegments(1, 27),  # Assuming default
            "OT": RegionSegments(28, 44),
            "SB": RegionSegments(45, 64),
        },
    ),

    ("aging", 1358, 64): KneeVideoMeta(
        condition="aging",
        video_id=1358,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(1360, 1384), FrameRange.from_1based(1385, 1406)),
            Cycle(FrameRange.from_1based(1407, 1433), FrameRange.from_1based(1434, 1454)),
            Cycle(FrameRange.from_1based(1461, 1483), FrameRange.from_1based(1484, 1508)),
            Cycle(FrameRange.from_1based(1509, 1540), FrameRange.from_1based(1541, 1559)),
            Cycle(FrameRange.from_1based(1618, 1648), FrameRange.from_1based(1649, 1669)),
            Cycle(FrameRange.from_1based(1672, 1696), FrameRange.from_1based(1697, 1720)), 
        ],
        regions={
            "JC": RegionSegments(1, 27),  # Assuming default
            "OT": RegionSegments(28, 44),
            "SB": RegionSegments(45, 64),
        },
    ),

    # Aging knees, N=16 (cycles = N=64; JC/OT/SB downscaled from 64-seg ranges)
    ("aging", 1339, 16): KneeVideoMeta(
        condition="aging",
        video_id=1339,
        n_segments=16,
        cycles=[  # same frames as ("aging", 1339, 64)
            Cycle(FrameRange.from_1based(290, 309), FrameRange.from_1based(312, 329)),
            Cycle(FrameRange.from_1based(331, 352), FrameRange.from_1based(355, 374)),
            Cycle(FrameRange.from_1based(375, 394), FrameRange.from_1based(398, 421)),
            Cycle(FrameRange.from_1based(422, 439), FrameRange.from_1based(441, 463)),
            Cycle(FrameRange.from_1based(464, 488), FrameRange.from_1based(490, 512)),
            Cycle(FrameRange.from_1based(513, 530), FrameRange.from_1based(532, 553)),
            Cycle(FrameRange.from_1based(554, 576), FrameRange.from_1based(579, 609)),
        ],
        regions={
            "JC": RegionSegments(1, 7),
            "OT": RegionSegments(8, 11),
            "SB": RegionSegments(12, 16),
        },
    ),

    ("aging", 1342, 16): KneeVideoMeta(
        condition="aging",
        video_id=1342,
        n_segments=16,
        cycles=[  # same 7 cycles as the updated ("aging", 1342, 64)
            Cycle(FrameRange.from_1based(62, 81),  FrameRange.from_1based(82, 100)),
            Cycle(FrameRange.from_1based(102, 119), FrameRange.from_1based(123, 151)),
            Cycle(FrameRange.from_1based(152, 171), FrameRange.from_1based(178, 199)),
            Cycle(FrameRange.from_1based(206, 222), FrameRange.from_1based(223, 246)),
            Cycle(FrameRange.from_1based(247, 272), FrameRange.from_1based(273, 297)),
            Cycle(FrameRange.from_1based(298, 320), FrameRange.from_1based(321, 340)),
            Cycle(FrameRange.from_1based(341, 364), FrameRange.from_1based(365, 384)),
        ],
        regions={
            "JC": RegionSegments(1, 7),
            "OT": RegionSegments(8, 11),
            "SB": RegionSegments(12, 16),
        },
    ),

    # DMM knees, N=64
    ("dmm-0w", 1195, 64): KneeVideoMeta(
        condition="dmm-0w",
        video_id=1195,
        n_segments=64,
        cycles=[
            Cycle(FrameRange.from_1based(771, 849), FrameRange.from_1based(855, 922)),
            Cycle(FrameRange.from_1based(929, 988), FrameRange.from_1based(989, 1047)),
        ],
        regions={
            "JC": RegionSegments(1, 27),
            "OT": RegionSegments(28, 40),
            "SB": RegionSegments(41, 64),
        },
    ),

    # DMM knees, N=64 (additional stub for 4w; cycles to be filled manually)
    # ("dmm-4w", 91, 64): KneeVideoMeta(
    #     condition="dmm-4w",
    #     video_id=91,
    #     n_segments=64,
    #     cycles=[
    #         # TODO: fill flexion/extension frame ranges, e.g.
    #         # Cycle(FrameRange.from_1based(start_flex, end_flex), FrameRange.from_1based(start_ext, end_ext)),
    #     ],
    #     regions={
    #         "JC": RegionSegments(1, 27),   # provisional: copied from dmm-0w 1195
    #         "OT": RegionSegments(28, 40),
    #         "SB": RegionSegments(41, 64),
    #     },
    # ),
}


def get_knee_meta(condition: str, video_id: int, n_segments: int) -> KneeVideoMeta:
    key = (condition, int(video_id), n_segments)
    try:
        return KNEE_VIDEOS[key]
    except KeyError:
        raise KeyError(f"No metadata for {key=}")
