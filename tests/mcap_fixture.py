from __future__ import annotations

from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2
from mcap.writer import Writer

HEVC_AU = b"\x00\x00\x00\x01\x26\x01\xaa"
F = descriptor_pb2.FieldDescriptorProto


def _field(
    message: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    *,
    repeated: bool = False,
    type_name: str = "",
) -> None:
    field = message.field.add(name=name, number=number, type=field_type)
    field.label = F.LABEL_REPEATED if repeated else F.LABEL_OPTIONAL
    if type_name:
        field.type_name = type_name


def descriptor_set_bytes() -> bytes:
    calibration = descriptor_pb2.FileDescriptorProto(
        name="calibration.proto", package="autonome", syntax="proto3"
    )
    intrinsic = calibration.message_type.add(name="CameraIntrinsic")
    for number, name in enumerate(
        [
            "focal_length_x",
            "focal_length_y",
            "optical_center_x",
            "optical_center_y",
            "rmse",
            "skew",
        ],
        1,
    ):
        _field(intrinsic, name, number, F.TYPE_DOUBLE)
    _field(intrinsic, "distortion_coeffs", 7, F.TYPE_DOUBLE, repeated=True)
    _field(intrinsic, "width", 8, F.TYPE_INT32)
    _field(intrinsic, "height", 9, F.TYPE_INT32)
    extrinsic = calibration.message_type.add(name="CameraExtrinsic")
    _field(extrinsic, "rotation_vector", 1, F.TYPE_DOUBLE, repeated=True)
    _field(extrinsic, "translation_vector", 2, F.TYPE_DOUBLE, repeated=True)

    for type_name, fields in {
        "LatLonHt": ["latitude_deg", "longitude_deg", "height_m"],
        "Orientation": ["roll_rad", "pitch_rad", "yaw_rad"],
        "PositionError": ["east_sigma_m", "north_sigma_m", "up_sigma_m", "hdop"],
        "OrientationError": ["roll_variance", "pitch_variance", "yaw_variance"],
    }.items():
        nested = calibration.message_type.add(name=type_name)
        for number, name in enumerate(fields, 1):
            _field(nested, name, number, F.TYPE_DOUBLE)

    telemetry = descriptor_pb2.FileDescriptorProto(
        name="telemetry.proto",
        package="autonome",
        syntax="proto3",
        dependency=["google/protobuf/timestamp.proto", "calibration.proto"],
    )
    camera = telemetry.message_type.add(name="CompressedVideos")
    _field(camera, "rec_frame_id", 1, F.TYPE_INT64)
    _field(camera, "rec_timestamp", 2, F.TYPE_MESSAGE, type_name=".google.protobuf.Timestamp")
    _field(camera, "timestamp", 3, F.TYPE_MESSAGE, type_name=".google.protobuf.Timestamp")
    _field(camera, "format", 4, F.TYPE_STRING)
    _field(camera, "frame_id", 5, F.TYPE_INT64)
    _field(camera, "data", 6, F.TYPE_BYTES, repeated=True)
    _field(camera, "name", 7, F.TYPE_STRING, repeated=True)
    _field(camera, "width", 8, F.TYPE_INT32)
    _field(camera, "height", 9, F.TYPE_INT32)
    _field(camera, "number_of_cameras", 10, F.TYPE_INT32)
    _field(
        camera,
        "camera_intrinsic",
        11,
        F.TYPE_MESSAGE,
        repeated=True,
        type_name=".autonome.CameraIntrinsic",
    )
    _field(
        camera,
        "camera_extrinsic",
        12,
        F.TYPE_MESSAGE,
        repeated=True,
        type_name=".autonome.CameraExtrinsic",
    )
    _field(
        camera,
        "camera_timestamp",
        14,
        F.TYPE_MESSAGE,
        repeated=True,
        type_name=".google.protobuf.Timestamp",
    )
    gnss = telemetry.message_type.add(name="GnssFix")
    _field(gnss, "timestamp", 1, F.TYPE_MESSAGE, type_name=".google.protobuf.Timestamp")
    _field(gnss, "rec_timestamp", 2, F.TYPE_MESSAGE, type_name=".google.protobuf.Timestamp")
    _field(gnss, "is_valid", 3, F.TYPE_BOOL)
    for number, (name, type_name) in enumerate(
        [
            ("lat_lon_ht", "LatLonHt"),
            ("orientation", "Orientation"),
            ("position_error", "PositionError"),
            ("orientation_error", "OrientationError"),
        ],
        4,
    ):
        _field(gnss, name, number, F.TYPE_MESSAGE, type_name=f".autonome.{type_name}")
    _field(gnss, "receiver_id", 8, F.TYPE_STRING)

    # Deliberately reverse dependency order to exercise the dynamic loader.
    file_set = descriptor_pb2.FileDescriptorSet(file=[telemetry, calibration])
    return file_set.SerializeToString()


def message_classes(descriptor_data: bytes | None = None) -> tuple[type[Any], type[Any]]:
    files = descriptor_pb2.FileDescriptorSet.FromString(
        descriptor_data or descriptor_set_bytes()
    )
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)  # type: ignore[no-untyped-call]
    pool.Add(files.file[1])  # type: ignore[no-untyped-call]
    pool.Add(files.file[0])  # type: ignore[no-untyped-call]
    return (
        message_factory.GetMessageClass(
            pool.FindMessageTypeByName("autonome.CompressedVideos")  # type: ignore[no-untyped-call]
        ),
        message_factory.GetMessageClass(
            pool.FindMessageTypeByName("autonome.GnssFix")  # type: ignore[no-untyped-call]
        ),
    )


def _timestamp(target: Any, timestamp_ns: int) -> None:
    target.seconds = timestamp_ns // 1_000_000_000
    target.nanos = timestamp_ns % 1_000_000_000


def camera_message(
    rec_timestamp_ns: int = 1_000_000_000,
    camera_timestamps_ns: tuple[int, ...] = (1_000_000_010, 1_000_000_020),
    *,
    format_name: str = "h265",
    payloads: tuple[bytes, ...] | None = None,
    dimensions: tuple[int, int] = (4, 3),
) -> bytes:
    camera_type, _ = message_classes()
    message = camera_type()
    message.rec_frame_id = 9
    message.frame_id = 10
    _timestamp(message.rec_timestamp, rec_timestamp_ns)
    _timestamp(message.timestamp, rec_timestamp_ns + 5)
    message.format = format_name
    message.width, message.height = dimensions
    message.number_of_cameras = 2
    frame_payloads = payloads or tuple(HEVC_AU for _ in camera_timestamps_ns)
    for index, timestamp_ns in enumerate(camera_timestamps_ns):
        message.data.append(frame_payloads[index])
        message.name.append(f"cam_{index}")
        _timestamp(message.camera_timestamp.add(), timestamp_ns)
        intrinsic = message.camera_intrinsic.add(
            focal_length_x=1,
            focal_length_y=1,
            optical_center_x=2,
            optical_center_y=2,
            width=dimensions[0],
            height=dimensions[1],
        )
        intrinsic.distortion_coeffs.extend([0.1, 0.2])
        extrinsic = message.camera_extrinsic.add()
        extrinsic.rotation_vector.extend([0, 0, 0])
        extrinsic.translation_vector.extend([index, 0, 0])
    return bytes(message.SerializeToString())


def gnss_message(timestamp_ns: int, longitude: float) -> bytes:
    _, gnss_type = message_classes()
    message = gnss_type(is_valid=True, receiver_id="rx-1")
    _timestamp(message.timestamp, timestamp_ns)
    _timestamp(message.rec_timestamp, timestamp_ns + 1)
    message.lat_lon_ht.latitude_deg = 0
    message.lat_lon_ht.longitude_deg = longitude
    message.lat_lon_ht.height_m = 100 + longitude
    message.orientation.yaw_rad = longitude / 100
    message.position_error.east_sigma_m = 0.1 + longitude
    message.position_error.north_sigma_m = 0.2
    message.position_error.up_sigma_m = 0.3
    message.position_error.hdop = 1
    message.orientation_error.yaw_variance = 0.01 + longitude
    return bytes(message.SerializeToString())


def write_mcap(
    path: Path,
    *,
    camera_schema_name: str = "autonome.CompressedVideos",
    camera_encoding: str = "protobuf",
    camera_payloads: tuple[bytes, ...] | None = None,
    include_camera: bool = True,
    include_gnss: bool = True,
    descriptor_data: bytes | None = None,
    gnss_payloads: tuple[bytes, ...] | None = None,
) -> None:
    schema_data = descriptor_data or descriptor_set_bytes()
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        if include_camera:
            schema = writer.register_schema(camera_schema_name, camera_encoding, schema_data)
            channel = writer.register_channel("rec_cameras", "protobuf", schema)
            payloads = camera_payloads or (camera_message(),)
            for index, payload in enumerate(payloads):
                writer.add_message(channel, (index + 1) * 100, payload, (index + 1) * 100)
        if include_gnss:
            schema = writer.register_schema("autonome.GnssFix", "protobuf", schema_data)
            channel = writer.register_channel("gnss", "protobuf", schema)
            payloads = gnss_payloads or (
                gnss_message(900_000_000, 0),
                gnss_message(2_100_000_000, 2),
            )
            for index, payload in enumerate(payloads):
                log_time = 50 + index * 200
                writer.add_message(channel, log_time, payload, log_time)
        writer.finish()
