from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OmniComponentPaths:
    sensevoice: str = "models/minimind-o/SenseVoiceSmall"
    siglip: str = "models/minimind-o/siglip2-base-p32-256-ve"
    mimi: str = "models/minimind-o/mimi"
    speaker_embeddings: str = "models/minimind-o/speaker/voices_unseen.pt"


@dataclass
class OmniSessionState:
    text_tokens: list[int] = field(default_factory=list)
    audio_frames: list[tuple[int, ...]] = field(default_factory=list)
    generating: bool = False


class OmniSession:
    def __init__(self, paths: OmniComponentPaths | None = None) -> None:
        self.paths = paths or OmniComponentPaths()
        self.state = OmniSessionState()

    def reset(self) -> None:
        self.state = OmniSessionState()

    def push_text_tokens(self, tokens: list[int]) -> None:
        self.state.text_tokens.extend(tokens)

    def push_audio_frame(self, frame: tuple[int, ...]) -> None:
        if len(frame) != 8:
            raise ValueError("MiniMind-O audio frames must contain 8 Mimi code streams")
        self.state.audio_frames.append(frame)
