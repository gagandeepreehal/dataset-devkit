from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from dataset_devkit.extraction.camera import (
    AssociatedDecodedFrame,
    CameraDecoderSet,
    DecoderOutput,
    PyAvHevcDecoder,
)
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.service import RecordingExtractor
from mcap_fixture import HEVC_AU, camera_message, write_mcap


class DeterministicDecoder:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def decode(
        self, payload: bytes, pts: int, time_base: Fraction
    ) -> list[DecoderOutput]:
        assert payload == HEVC_AU
        assert time_base == Fraction(1, 1_000_000_000)
        self.calls += 1
        return [DecoderOutput(pts, Image.new("RGB", (4, 3), (self.calls, 2, 3)))]

    def flush(self) -> list[DecoderOutput]:
        return []

    def close(self) -> None:
        self.closed = True


class OneFrameDelayedDecoder:
    def __init__(self) -> None:
        self.pending: DecoderOutput | None = None
        self.closed = False

    def decode(
        self, payload: bytes, pts: int, time_base: Fraction
    ) -> list[DecoderOutput]:
        emitted = [] if self.pending is None else [self.pending]
        self.pending = DecoderOutput(pts, Image.new("RGB", (4, 3), (pts + 1, 0, 0)))
        return emitted

    def flush(self) -> list[DecoderOutput]:
        return [] if self.pending is None else [self.pending]

    def close(self) -> None:
        self.closed = True


def test_end_to_end_recording_uses_real_camera_timestamps_for_images_and_poses(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.mcap"
    write_mcap(
        path,
        camera_payloads=(
            camera_message(1_000_000_000, (1_000_000_010, 1_000_000_020)),
            camera_message(1_500_000_000, (1_500_000_010, 1_500_000_020)),
        ),
    )
    decoders: list[DeterministicDecoder] = []

    def factory() -> DeterministicDecoder:
        decoder = DeterministicDecoder()
        decoders.append(decoder)
        return decoder

    result = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=Fraction(2, 1),
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=factory,
    ).extract(path)

    assert len(result.samples) == 4
    assert [sample.camera_timestamp_ns for sample in result.samples] == [
        1_000_000_010,
        1_000_000_020,
        1_500_000_010,
        1_500_000_020,
    ]
    assert all(
        sample.staged_image.timestamp_ns == sample.camera_timestamp_ns
        for sample in result.samples
    )
    assert all(
        sample.ego_pose.timestamp_ns == sample.camera_timestamp_ns
        for sample in result.samples
    )
    assert len(result.ego_poses_by_timestamp) == 4
    assert [decoder.calls for decoder in decoders] == [2, 2]
    assert all(decoder.closed for decoder in decoders)
    assert all(sample.staged_image.path.is_file() for sample in result.samples)
    assert all(
        not hasattr(frame, "payload")
        for batch in result.camera_batches
        for frame in batch.frames
    )
    with pytest.raises(TypeError):
        result.ego_poses_by_timestamp[0] = result.samples[0].ego_pose  # type: ignore[index]


@pytest.mark.parametrize(
    ("rec_timestamps", "expected_batches", "expected_misses"),
    [
        (
            (1_000_000_000, 1_490_000_000, 2_010_000_000),
            (1_000_000_000, 1_490_000_000, 2_010_000_000),
            (),
        ),
        (
            (1_000_000_000, 2_000_000_000),
            (1_000_000_000, 2_000_000_000),
            (1_500_000_000,),
        ),
    ],
)
def test_actual_mcap_jitter_and_dropped_batch_grid_scenarios(
    tmp_path: Path,
    rec_timestamps: tuple[int, ...],
    expected_batches: tuple[int, ...],
    expected_misses: tuple[int, ...],
) -> None:
    path = tmp_path / "scenario.mcap"
    payloads = tuple(
        camera_message(
            timestamp,
            (timestamp + 10, timestamp + 20),
        )
        for timestamp in rec_timestamps
    )
    write_mcap(path, camera_payloads=payloads)

    result = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=Fraction(2, 1),
        tolerance_ns=20_000_000,
        staging_root=tmp_path / "staging",
        decoder_factory=DeterministicDecoder,
    ).extract(path)

    assert tuple(entry.batch_timestamp_ns for entry in result.selected_grid.entries) == (
        expected_batches
    )
    assert tuple(miss.target_timestamp_ns for miss in result.selected_grid.misses) == (
        expected_misses
    )


def test_delayed_outputs_stage_only_their_originating_selected_access_units(
    tmp_path: Path,
) -> None:
    path = tmp_path / "delayed-selected.mcap"
    write_mcap(
        path,
        camera_payloads=tuple(
            camera_message(timestamp, (timestamp + 10, timestamp + 20))
            for timestamp in (1_000_000_000, 1_500_000_000, 2_000_000_000)
        ),
    )

    result = RecordingExtractor(
        camera_topic="rec_cameras",
        gnss_topic="gnss",
        target_fps=Fraction(1, 1),
        tolerance_ns=0,
        staging_root=tmp_path / "staging",
        decoder_factory=OneFrameDelayedDecoder,
    ).extract(path)

    assert [sample.batch_timestamp_ns for sample in result.samples] == [
        1_000_000_000,
        1_000_000_000,
        2_000_000_000,
        2_000_000_000,
    ]
    assert all(result.staging_root in sample.staged_image.path.parents for sample in result.samples)
    assert len({sample.staged_image.path.name for sample in result.samples}) == 4


def _inter_frame_hevc_access_units() -> list[bytes]:
    av = pytest.importorskip("av", reason="PyAV is unavailable for real HEVC regression")
    np = pytest.importorskip("numpy", reason="NumPy is unavailable for real HEVC regression")
    try:
        encoder = av.CodecContext.create("libx265", "w")
        encoder.width = 32
        encoder.height = 32
        encoder.pix_fmt = "yuv420p"
        encoder.time_base = Fraction(1, 30)
        encoder.framerate = Fraction(30, 1)
        encoder.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "x265-params": "keyint=30:min-keyint=30:scenecut=0:bframes=0:repeat-headers=1",
        }
        encoder.open()
    except Exception as error:
        pytest.skip(f"local libx265 encoder unavailable: {error}")
    packets: list[bytes] = []
    for index in range(2):
        pixels = np.full((32, 32, 3), 50 + 40 * index, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = index
        packets.extend(bytes(packet) for packet in encoder.encode(frame))
    packets.extend(bytes(packet) for packet in encoder.encode(None))
    if len(packets) != 2:
        pytest.skip(f"local libx265 did not emit one access unit per input frame: {len(packets)}")
    return packets


def test_real_pyav_decoder_keeps_state_for_inter_frame_hevc() -> None:
    access_units = _inter_frame_hevc_access_units()
    decoder = CameraDecoderSet(1, PyAvHevcDecoder)
    outputs = [
        *decoder.submit(0, access_units[0], "first"),
        *decoder.submit(0, access_units[1], "second"),
        *decoder.finish(),
    ]
    assert sorted(output.metadata for output in outputs) == ["first", "second"]
    assert all(output.image.size == (32, 32) for output in outputs)

    fresh = CameraDecoderSet(1, PyAvHevcDecoder)
    with pytest.raises(StructuralExtractionError):
        fresh.submit(0, access_units[1], "second")
        fresh.finish()


def _b_frame_hevc_access_units() -> list[bytes]:
    av = pytest.importorskip("av", reason="PyAV is unavailable for B-frame regression")
    np = pytest.importorskip("numpy", reason="NumPy is unavailable for B-frame regression")
    try:
        encoder = av.CodecContext.create("libx265", "w")
        encoder.width = 32
        encoder.height = 32
        encoder.pix_fmt = "yuv420p"
        encoder.time_base = Fraction(1, 30)
        encoder.framerate = Fraction(30, 1)
        encoder.options = {
            "preset": "ultrafast",
            "x265-params": (
                "keyint=30:min-keyint=30:scenecut=0:bframes=3:repeat-headers=1"
            ),
        }
        encoder.open()
    except Exception as error:
        pytest.skip(f"local libx265 B-frame encoder unavailable: {error}")
    packets: list[bytes] = []
    for index in range(10):
        pixels = np.full((32, 32, 3), 10 + 20 * index, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = index
        packets.extend(bytes(packet) for packet in encoder.encode(frame))
    packets.extend(bytes(packet) for packet in encoder.encode(None))
    if len(packets) != 10:
        pytest.skip(f"local libx265 emitted {len(packets)} packets for 10 B-frame inputs")
    return packets


def test_real_pyav_b_frames_are_associated_by_pts_and_flushed_at_eof() -> None:
    access_units = _b_frame_hevc_access_units()
    decoder = CameraDecoderSet(1, PyAvHevcDecoder)
    emitted: list[AssociatedDecodedFrame] = []
    per_submit_counts = []
    for index, access_unit in enumerate(access_units):
        current = decoder.submit(0, access_unit, index)
        per_submit_counts.append(len(current))
        emitted.extend(current)
    flushed = decoder.finish()
    emitted.extend(flushed)

    assert 0 in per_submit_counts
    assert flushed
    assert sorted(output.metadata for output in emitted) == list(range(10))
    assert len({output.submission_index for output in emitted}) == 10


def test_real_reader_service_rolls_back_staged_files_after_undecodable_hevc(
    tmp_path: Path,
) -> None:
    access_units = _inter_frame_hevc_access_units()
    path = tmp_path / "corrupt-after-valid.mcap"
    write_mcap(
        path,
        camera_payloads=(
            camera_message(
                1_000_000_000,
                (1_000_000_010, 1_000_000_020),
                payloads=(access_units[0], access_units[0]),
                dimensions=(32, 32),
            ),
            camera_message(
                1_500_000_000,
                (1_500_000_010, 1_500_000_020),
                payloads=(access_units[1], HEVC_AU),
                dimensions=(32, 32),
            ),
        ),
    )
    staging_root = tmp_path / "staging"

    with pytest.raises(StructuralExtractionError, match="HEVC|decoded frame|missing.*frame"):
        RecordingExtractor(
            camera_topic="rec_cameras",
            gnss_topic="gnss",
            target_fps=Fraction(2, 1),
            tolerance_ns=0,
            staging_root=staging_root,
        ).extract(path)
    assert not list(staging_root.rglob("*.jpg"))
