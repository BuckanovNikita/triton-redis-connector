"""Pydantic event models for the multi-source enrichment pipeline.

Four distinct sensor event types are enriched by the processor into
four corresponding enriched types, preserving domain-specific fields.
"""

from typing import Annotated, ClassVar

from typed_flow.protocols import Counter, Gauge, GroupBy
from typed_flow.protocols.messaging import BaseEvent
from typed_flow.protocols.metrics import TimeMetric


class TemperatureReading(BaseEvent):
    """Temperature sensor event with Celsius value and sensor location."""

    _metrics_: ClassVar[list[Counter]] = [Counter("temperature_received")]

    sensor_id: str
    celsius: float
    location: str


class HumidityReading(BaseEvent):
    """Humidity sensor event with relative humidity percentage."""

    _metrics_: ClassVar[list[Counter]] = [Counter("humidity_received")]

    sensor_id: str
    humidity_pct: float
    zone: str


class PressureReading(BaseEvent):
    """Atmospheric pressure sensor event in hectopascals."""

    _metrics_: ClassVar[list[Counter]] = [Counter("pressure_received")]

    sensor_id: str
    hpa: float
    altitude_m: float


class WindReading(BaseEvent):
    """Wind sensor event with speed and direction."""

    _metrics_: ClassVar[list[Counter]] = [Counter("wind_received")]

    sensor_id: str
    speed_mps: float
    direction_deg: int
    gust_mps: float


class EnrichedTemperature(BaseEvent):
    """Enriched temperature reading with random score and processing timestamp."""

    _metrics_: ClassVar[list[Counter]] = [Counter("enriched_events")]

    sensor_id: str
    celsius: Annotated[float, Gauge("sensor_celsius")]
    location: str
    sensor_source: Annotated[str, GroupBy("events_by_source", window_s=60)]
    random_score: Annotated[float, Gauge("random_score")]
    processed_at: TimeMetric


class EnrichedHumidity(BaseEvent):
    """Enriched humidity reading with random score and processing timestamp."""

    _metrics_: ClassVar[list[Counter]] = [Counter("enriched_events")]

    sensor_id: str
    humidity_pct: Annotated[float, Gauge("sensor_humidity_pct")]
    zone: str
    sensor_source: Annotated[str, GroupBy("events_by_source", window_s=60)]
    random_score: Annotated[float, Gauge("random_score")]
    processed_at: TimeMetric


class EnrichedPressure(BaseEvent):
    """Enriched pressure reading with random score and processing timestamp."""

    _metrics_: ClassVar[list[Counter]] = [Counter("enriched_events")]

    sensor_id: str
    hpa: Annotated[float, Gauge("sensor_hpa")]
    altitude_m: float
    sensor_source: Annotated[str, GroupBy("events_by_source", window_s=60)]
    random_score: Annotated[float, Gauge("random_score")]
    processed_at: TimeMetric


class EnrichedWind(BaseEvent):
    """Enriched wind reading with random score and processing timestamp."""

    _metrics_: ClassVar[list[Counter]] = [Counter("enriched_events")]

    sensor_id: str
    speed_mps: Annotated[float, Gauge("sensor_speed_mps")]
    direction_deg: int
    gust_mps: float
    sensor_source: Annotated[str, GroupBy("events_by_source", window_s=60)]
    random_score: Annotated[float, Gauge("random_score")]
    processed_at: TimeMetric


SensorReading = TemperatureReading | HumidityReading | PressureReading | WindReading
EnrichedReading = EnrichedTemperature | EnrichedHumidity | EnrichedPressure | EnrichedWind
