"""lipreader — probabilistic, multi-person lip reading built on syrup.

    from lipreader import analyze
    result = analyze("clip.mp4", mode="visual", language="auto")
    print(result.to_dict()["segments"])
"""
from .schema import NOTICE, AnalysisResult, Options, PersonTrack, TranscriptSegment, Word  # noqa: F401

__version__ = "0.1.0"


def analyze(path, **options):
    from .pipeline import analyze as _analyze

    return _analyze(path, Options.from_dict(options))
