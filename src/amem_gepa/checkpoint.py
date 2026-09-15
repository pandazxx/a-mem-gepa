"""Resumable checkpointing for the eval loop (evaluate.py). A single
conversation replay (turn-by-turn add_note, ~2 LLM calls/turn) can run for
hours against a slow local model -- losing all of it to a Ctrl-C, crash, or
laptop sleep is expensive (observed: a 12+ hour run that never finished).

Two independent stores, both under results/<run_label>/ (gitignored, see
CLAUDE.md -- these are run artifacts, not durable state):

- CheckpointStore: one JSON file per conversation, rewritten after every
  turn during replay, holding a full snapshot of that conversation's
  AgenticMemorySystem.memories (via amem_adapter.snapshot_notes/restore_notes)
  plus how many turns have been processed. On resume, a conversation's
  checkpoint is loaded and replay continues from turns_processed onward --
  at most one turn of work is ever lost, not a whole conversation.
- PredictionStore: an append-only JSONL log of (conversation_id, question)
  -> prediction, so already-answered questions are skipped on resume too.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConversationCheckpoint:
    conversation_id: str
    turns_processed: int = 0
    notes: list = field(default_factory=list)
    completed: bool = False


class CheckpointStore:
    def __init__(self, run_label: str, results_dir: Path = Path("results")):
        self.dir = Path(results_dir) / run_label / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        return self.dir / f"{conversation_id}.json"

    def load(self, conversation_id: str) -> Optional[ConversationCheckpoint]:
        path = self._path(conversation_id)
        if not path.exists():
            return None
        return ConversationCheckpoint(**json.loads(path.read_text()))

    def save(self, checkpoint: ConversationCheckpoint) -> None:
        path = self._path(checkpoint.conversation_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(asdict(checkpoint)))
        tmp_path.replace(path)  # atomic-ish: never leaves a half-written checkpoint


class PredictionStore:
    """Keyed on (conversation_id, question) -- good enough since LoCoMo
    questions are unique per conversation in the released dataset; not
    trying to handle duplicate question text within one conversation."""

    def __init__(self, run_label: str, split: str, results_dir: Path = Path("results")):
        self.path = Path(results_dir) / run_label / f"{split}_predictions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: dict[tuple[str, str], dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._done[(record["conversation_id"], record["question"])] = record

    def __len__(self) -> int:
        return len(self._done)

    def get(self, conversation_id: str, question: str) -> Optional[dict]:
        return self._done.get((conversation_id, question))

    def append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._done[(record["conversation_id"], record["question"])] = record
