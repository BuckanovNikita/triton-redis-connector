"""Pydantic event models for the multi-source merger pipeline.

Four distinct sensor event types converge into a single MergedReading.
Each input type has unique fields reflecting its sensor domain.
"""

from __future__ import annotations

from typed_flow.protocols.messaging import BaseEvent


class TemperatureReading(BaseEvent):
    """Temperature sensor event with Celsius value and sensor location."""

    sensor_id: str
    celsius: float
    location: str


class HumidityReading(BaseEvent):
    """Humidity sensor event with relative humidity percentage."""

    sensor_id: str
    humidity_pct: float
    zone: str


class PressureReading(BaseEvent):
    """Atmospheric pressure sensor event in hectopascals."""

    sensor_id: str
    hpa: float
    altitude_m: float


class WindReading(BaseEvent):
    """Wind sensor event with speed and direction."""

    sensor_id: str
    speed_mps: float
    direction_deg: int
    gust_mps: float


class MergedReading(BaseEvent):
    """Unified output carrying normalized sensor data into the merged sink."""

    sensor_id: str
    sensor_type: str
    value: float
    label: str
