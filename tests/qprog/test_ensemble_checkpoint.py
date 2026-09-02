# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ensemble checkpoint state and directory resolution."""

import json

import pytest
from pydantic import ValidationError

from divi.qprog._ensemble_checkpoint import (
    ROUND_COMPLETION_FILE,
    ROUND_START_FILE,
    ProgramRoundRecord,
    RoundCheckpoint,
    _encode_program_id,
    _program_checkpoint_path,
    _resolve_ensemble_checkpoint,
    _round_dir,
)
from divi.qprog._program_checkpoint import ProgramCheckpoint
from divi.qprog.checkpointing import CheckpointNotFoundError


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _interrupted_payload(round_index=1):
    return {
        "kind": "round_start",
        "ensemble_type": "tests.SampleEnsemble",
        "round_index": round_index,
        "ensemble_state": {},
        "programs": [],
    }


def _completed_payload(round_index=1):
    return {
        "kind": "round_completion",
        "ensemble_type": "tests.SampleEnsemble",
        "round_index": round_index,
        "ensemble_state": {},
        "round_history": [],
        "total_circuit_count": 0,
        "total_run_time": 0.0,
    }


class TestProgramIdEncoding:
    def test_encodes_nested_tuple_without_losing_tuple_identity(self):
        encoded = _encode_program_id(("ansatz", (2, None)))

        assert encoded == {
            "kind": "tuple",
            "items": [
                {"kind": "str", "value": "ansatz"},
                {
                    "kind": "tuple",
                    "items": [
                        {"kind": "int", "value": 2},
                        {"kind": "none"},
                    ],
                },
            ],
        }

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
    def test_rejects_values_without_stable_json_identity(self, value):
        with pytest.raises(TypeError, match="checkpoint program ID"):
            _encode_program_id(value)


class TestStateModels:
    def test_child_recovery_state_has_no_computation_fingerprint(self):
        assert "identity" not in ProgramRoundRecord.model_fields

    def test_round_checkpoint_uses_an_explicit_kind(self):
        interrupted = RoundCheckpoint(
            kind="round_start",
            ensemble_type="tests.SampleEnsemble",
            round_index=1,
            ensemble_state={},
            programs=[],
        )
        completed = RoundCheckpoint(
            kind="round_completion",
            ensemble_type="tests.SampleEnsemble",
            round_index=1,
            ensemble_state={},
            round_history=[],
            total_circuit_count=0,
            total_run_time=0.0,
        )

        assert interrupted.kind == "round_start"
        assert completed.kind == "round_completion"

    @pytest.mark.parametrize(
        "kind_fields",
        [
            {"kind": "round_start"},
            {
                "kind": "round_completion",
                "programs": [],
                "round_history": [],
                "total_circuit_count": 0,
                "total_run_time": 0.0,
            },
        ],
    )
    def test_round_checkpoint_rejects_an_incomplete_or_mixed_kind(self, kind_fields):
        with pytest.raises(ValidationError):
            RoundCheckpoint(
                ensemble_type="tests.SampleEnsemble",
                round_index=1,
                ensemble_state={},
                **kind_fields,
            )

    def test_interrupted_round_round_trips(self):
        state = RoundCheckpoint(
            kind="round_start",
            ensemble_type="tests.SampleEnsemble",
            round_index=2,
            ensemble_state={"value": 3},
            programs=[
                ProgramRoundRecord(
                    program_id=_encode_program_id(("a", 1)),
                    program_type="tests.SampleProgram",
                    circuit_count_at_round_start=0,
                    run_time_at_round_start=0.0,
                )
            ],
        )

        assert RoundCheckpoint.model_validate_json(state.model_dump_json()) == state
        assert state.model_dump(mode="json")["programs"] == [
            {
                "program_id": _encode_program_id(("a", 1)),
                "program_type": "tests.SampleProgram",
                "circuit_count_at_round_start": 0,
                "run_time_at_round_start": 0.0,
            }
        ]

    def test_completed_round_rejects_negative_counters(self):
        with pytest.raises(ValidationError):
            RoundCheckpoint(
                kind="round_completion",
                ensemble_type="tests.SampleEnsemble",
                round_index=1,
                ensemble_state={},
                round_history=[],
                total_circuit_count=-1,
                total_run_time=0.0,
            )

    def test_completed_child_rejects_invalid_accounting(self):
        with pytest.raises(ValidationError):
            ProgramCheckpoint(
                program_type="tests.SampleProgram",
                total_circuit_count=-1,
                total_run_time=float("inf"),
            )

    def test_interrupted_round_requires_child_accounting(self):
        with pytest.raises(ValidationError):
            RoundCheckpoint(
                kind="round_start",
                ensemble_type="tests.SampleEnsemble",
                round_index=1,
                ensemble_state={},
                programs=[
                    ProgramRoundRecord(
                        program_id=_encode_program_id("a"),
                        program_type="tests.SampleProgram",
                    )
                ],
            )

    def test_program_checkpoint_path_uses_round_local_slot(self, tmp_path):
        assert _program_checkpoint_path(tmp_path, 2) == tmp_path / "program_002"


class TestRoundResolution:
    def test_round_dir_is_zero_padded(self, tmp_path):
        assert _round_dir(tmp_path, 7) == tmp_path / "round_007"

    def test_prefers_completed_marker_within_latest_round(self, tmp_path):
        round_dir = _round_dir(tmp_path, 2)
        _write_json(round_dir / ROUND_START_FILE, _interrupted_payload(2))
        _write_json(round_dir / ROUND_COMPLETION_FILE, _completed_payload(2))

        checkpoint = _resolve_ensemble_checkpoint(tmp_path)

        assert checkpoint.round_index == 2
        assert checkpoint.kind == "round_completion"

    def test_latest_interrupted_round_wins_over_earlier_completed_round(self, tmp_path):
        _write_json(
            _round_dir(tmp_path, 1) / ROUND_COMPLETION_FILE,
            _completed_payload(1),
        )
        _write_json(
            _round_dir(tmp_path, 2) / ROUND_START_FILE,
            _interrupted_payload(2),
        )

        checkpoint = _resolve_ensemble_checkpoint(tmp_path)

        assert checkpoint.round_index == 2
        assert checkpoint.kind == "round_start"

    def test_skips_higher_round_with_invalid_json(self, tmp_path):
        _write_json(
            _round_dir(tmp_path, 1) / ROUND_COMPLETION_FILE,
            _completed_payload(1),
        )
        invalid = _round_dir(tmp_path, 2) / ROUND_START_FILE
        invalid.parent.mkdir(parents=True)
        invalid.write_text("not-json")

        resolved = _resolve_ensemble_checkpoint(tmp_path)

        assert resolved.round_index == 1
        assert resolved.kind == "round_completion"

    def test_resolves_explicit_subdirectory(self, tmp_path):
        _write_json(
            _round_dir(tmp_path, 3) / ROUND_COMPLETION_FILE,
            _completed_payload(3),
        )

        resolved = _resolve_ensemble_checkpoint(tmp_path, "round_003")

        assert resolved.round_index == 3

    def test_completed_manual_snapshot_does_not_require_open_marker(self, tmp_path):
        _write_json(
            _round_dir(tmp_path, 1) / ROUND_COMPLETION_FILE,
            _completed_payload(1),
        )

        resolved = _resolve_ensemble_checkpoint(tmp_path)

        assert resolved.kind == "round_completion"

    def test_skips_marker_whose_state_artifact_is_missing(self, tmp_path):
        payload = _completed_payload(2)
        payload["ensemble_state"] = {"artifact": "output_state.npz"}
        _write_json(
            _round_dir(tmp_path, 2) / ROUND_COMPLETION_FILE,
            payload,
        )
        _write_json(
            _round_dir(tmp_path, 1) / ROUND_COMPLETION_FILE,
            _completed_payload(1),
        )

        resolved = _resolve_ensemble_checkpoint(tmp_path)

        assert resolved.round_index == 1

    def test_no_valid_round_raises(self, tmp_path):
        with pytest.raises(CheckpointNotFoundError, match="ensemble checkpoint"):
            _resolve_ensemble_checkpoint(tmp_path)
