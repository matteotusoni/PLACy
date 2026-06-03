import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_discovery.placy import PLACy as _PLACy


def PLACy(data, max_lags=20, window_length=50, stride=1, use_bh_fdr = False, acyclicity = False):
    return _PLACy(
        data,
        max_lags=max_lags,
        window_length=window_length,
        stride=stride,
        use_bh_fdr = use_bh_fdr,
        acyclicity = acyclicity,
    )