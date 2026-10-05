import pytest
from dataclasses import dataclass, field, FrozenInstanceError
from pydantic import BaseModel, Field, ValidationError, model_validator

# --- Classes ---
# Q1
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = [] if readings is None else list(readings)
# append a temp to the list
    def add(self, reading):
        self.readings.append(reading)
# divide the total by the number of readings
    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)
# find the largest reading
    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)
    
# Q2
    def __repr__(self):
        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )
    
# Create an object
my_thermometer = Thermometer("Chicago")

my_thermometer.add(18.0)
my_thermometer.add(22.0)
my_thermometer.add(16.0)
my_thermometer.add(24.0)

print("Average:", my_thermometer.average())
print("Hottest:", my_thermometer.hottest())

# average() handles an empty list because dividing by its length,
# which would be zero, would raise ZeroDivisionError.

empty_thermometer = Thermometer("Boston")

print("Empty average:", empty_thermometer.average())
print("Empty hottest:", empty_thermometer.hottest())

# Show one object and a list containing two objects.
print(my_thermometer)
print([my_thermometer, empty_thermometer])

# Without a custom __repr__, Python displays the class name and
# a memory address, rather than the location and readings.
# That makes debugging harder because we cannot see the object's data.

# Q3
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]
    
lower_alert = TemperatureAlert(threshold=20.0)
higher_alert = TemperatureAlert(threshold=23.0)

print("Above 20 C:", lower_alert.breaches(my_thermometer))
print("Above 23 C:", higher_alert.breaches(my_thermometer))

# Q3 Q & A's
# The threshold belongs to the alert because it is reusable configuration.
# With twenty thermometers, we can use the same alert to check each one
# without supplying the threshold again on every call.



# --- Dataclasses, Type Hints, and Docstrings ---
# Q1

@dataclass(frozen=True)
class Station:
    """A weather station with its identifier, name, and geographic position.

    Latitude and longitude are in degrees; elevation is in meters.
    """

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station(
    station_id="CHI001",
    name="Chicago",
    latitude=41.88,
    longitude=-87.63,
    elevation=181.0,
)

station_b = Station(
    station_id="CHI001",
    name="Chicago",
    latitude=41.88,
    longitude=-87.63,
    elevation=181.0,
)

print("Station A:", station_a)
print("Station B:", station_b)
print("Same field values:", station_a == station_b)

# A dataclass generates __eq__, so two Station objects with identical
# field values compare equal. The original class had no custom __eq__,
# so two separate objects would compare unequal even with identical data.


# Q2
try:
    station_a.elevation = 200.0
except FrozenInstanceError as error:
    print("Cannot change station:", error)

station_c = Station(
    station_id="DEN001",
    name="Denver",
    latitude=39.74,
    longitude=-104.99,
    elevation=1609.0,
)

stations = {station_a, station_b, station_c}

print("Number of unique stations:", len(stations))

# With the default equality settings, frozen=True also generates a hash.
# These Station fields are hashable, so stations can be stored in a set.
# A set removes duplicates: station_a and station_b have identical
# values, so the three objects produce only two unique entries.



# Q3

# Original error:
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
# A default [] would be shared between instances. Dataclasses reject it
# to prevent batches from accidentally sharing the same list.
# default_factory=list creates a fresh list for each batch.

@dataclass
class StationBatch:
    """A collection of weather stations in a region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Append a station to this batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)

batch = StationBatch(region="United States")
batch.add(station_a)
batch.add(station_c)

print("Highest station:", batch.highest())

empty_batch = StationBatch(region="Empty region")
print("Highest in empty batch:", empty_batch.highest())
print("Empty batch station count:", len(empty_batch.stations))


# --- Pydantic ---
# Q1

class Reading(BaseModel):
    """A weather station's temperature and humidity measurement."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)
### Q4 ###
    @model_validator(mode="after")
    def check_sensor(self) -> "Reading":
        """Reject zero humidity combined with temperature below -40 C."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Possible failed sensor: zero humidity below -40 C")
        return self

valid_reading = Reading(
    station_id="CHI001",
    timestamp="2026-10-04T09:00:00",
    temperature_c=21.5,
    humidity=40.0,
)

print("Valid reading:", valid_reading)


# Q2

# Failure 1: missing the required timestamp.
try:
    Reading(
        station_id="CHI001",
        temperature_c=21.5,
        humidity=40.0,
    )
except ValidationError as error:
    print("\nMissing timestamp:")
    print(error)

# Failure 2: temperature exceeds the maximum of 60.
try:
    Reading(
        station_id="CHI001",
        timestamp="2026-10-04T09:00:00",
        temperature_c=150.0,
        humidity=40.0,
    )
except ValidationError as error:
    print("\nTemperature out of range:")
    print(error)

# Failure 3: humidity cannot be converted into a number.
try:
    Reading(
        station_id="CHI001",
        timestamp="2026-10-04T09:00:00",
        temperature_c=21.5,
        humidity="very humid",
    )
except ValidationError as error:
    print("\nNon-numeric humidity:")
    print(error)

converted_reading = Reading(
    station_id="CHI001",
    timestamp="2026-10-04T09:00:00",
    temperature_c="21.5",
    humidity=40,
)

print("\nConverted reading:", converted_reading)
print("Temperature type:", type(converted_reading.temperature_c))
print("Humidity type:", type(converted_reading.humidity))

# Pydantic accepts "21.5" because it can convert numeric text to a float.
# It rejects "very humid" because that text cannot be parsed as a number.


# Q3

try:
    Reading(
        station_id="AB",
        temperature_c="hot",
        humidity=40.0,
    )
except ValidationError as error:
    print("\nMultiple validation errors:")
    for problem in error.errors():
        print("Field:", problem["loc"], "| Message:", problem["msg"])

# Three errors are reported: station_id is too short, timestamp is
# missing, and temperature_c cannot be converted to a number.
# Reporting all three lets us fix the input in one pass instead of
# repeatedly running the program to discover the next problem.

# Q4 demonstration

still_valid = Reading(
    station_id="CHI001",
    timestamp="2026-10-04T09:00:00",
    temperature_c=-45.0,
    humidity=20.0,
)
print("\nValid cold reading:", still_valid)

try:
    Reading(
        station_id="CHI001",
        timestamp="2026-10-04T09:00:00",
        temperature_c=-45.0,
        humidity=0.0,
    )
except ValidationError as error:
    print("\nFailed sensor combination:")
    print(error)

# Field constraints check individual values against fixed limits.
# This rule depends on the combination of humidity and temperature,
# so it needs a model validator that can inspect both fields.

# --- Testing with pytest ---
# Q1

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)
# The plain == comparison passed with this calculation on my machine.
# Floating-point arithmetic can introduce tiny rounding differences.
# pytest.approx compares within a tolerance instead of requiring exact equality.


# Q2

def mean(values: list[float]) -> float:
    """Return the average; raise ValueError if values is empty."""
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])


# pytest.raises(ValueError) alone checks only the exception type.
# match="empty" also checks the message, helping confirm that the
# error describes the empty-input problem we intended to test.

# Q3

@pytest.mark.parametrize(
    "values, expected",
    [
        ([2.0, 4.0, 6.0], 4.0),
        ([7.0], 7.0),
        ([-6.0, -2.0], -4.0),
        ([-3.0, 0.0, 3.0], 0.0),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)


# Parametrize runs the same test with four input/output pairs.
# It avoids repeating the test logic, makes new cases easy to add,
# and reports each case separately so we can identify failures.
# Pytest summary: 6 passed in 0.16s


# Q4
# Deliberately changed the conversion denominator from 5 to 4.
# Failure output:
# E       assert 257.0 == 212
# E        + where 257.0 = celsius_to_fahrenheit(100)
# assignments_01\warmup_01.py:320: AssertionError
# FAILED assignments_01/warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# 1 failed, 5 passed in 0.34s
#
# Pytest showed the actual result, 257.0, and the expected result, 212,
# for the input 100. These values identify which calculation was wrong
# and how its result differed, unlike a bare "assertion failed" message.