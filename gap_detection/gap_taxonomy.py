from enum import Enum


class GapType(str, Enum):
    EXPLICIT = "explicit_gap"
    IMPLICIT = "implicit_gap"
    MISSING_LINK = "missing_link"
    CONTRADICTORY = "contradictory_gap"


GAP_TYPE_DESCRIPTIONS = {
    GapType.EXPLICIT: "Authors explicitly state this remains unknown or understudied",
    GapType.IMPLICIT: "Expected concept co-occurrence is statistically absent across clusters",
    GapType.MISSING_LINK: "A\u2192B and B\u2192C relationships exist but A\u2192C is unexplored",
    GapType.CONTRADICTORY: "Studies disagree on a mechanism or finding",
}
