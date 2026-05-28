from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from minimind_orangepi.omni_session import OmniSession


def test_omni_session_state_boundaries() -> None:
    session = OmniSession()
    session.push_text_tokens([1, 2, 3])
    session.push_audio_frame((1, 2, 3, 4, 5, 6, 7, 8))
    assert session.state.text_tokens == [1, 2, 3]
    assert len(session.state.audio_frames) == 1
    with pytest.raises(ValueError):
        session.push_audio_frame((1, 2, 3))
    session.reset()
    assert session.state.text_tokens == []
    assert session.state.audio_frames == []
