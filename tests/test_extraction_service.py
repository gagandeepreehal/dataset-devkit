from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from dataset_devkit.extraction.camera import CameraDecoderSet, PyAvHevcDecoder
from dataset_devkit.extraction.errors import StructuralExtractionError
from dataset_devkit.extraction.service import RecordingExtractor
from mcap_fixture import HEVC_AU, camera_message, write_mcap


class DeterministicDecoder:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def decode(self, payload: bytes) -> list[Image.Image]:
        assert payload == HEVC_AU
        self.calls += 1
        return [Image.new("RGB", (4, 3), (self.calls, 2, 3))]

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
    assert decoder.decode(0, access_units[0]).size == (32, 32)
    assert decoder.decode(0, access_units[1]).size == (32, 32)
    decoder.close()

    fresh = CameraDecoderSet(1, PyAvHevcDecoder)
    with pytest.raises(StructuralExtractionError):
        fresh.decode(0, access_units[1])
