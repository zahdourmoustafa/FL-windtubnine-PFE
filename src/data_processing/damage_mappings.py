"""
Mappings between sensors, components, and failure modes for the gearbox.
Used to translate sensor anomalies into specific component diagnoses.
"""

# Mapping from sensor labels to the components they monitor
SENSOR_TO_COMPONENT = {
    "AN3": "Ring gear (6 o'clock position)",
    "AN4": "Ring gear (12 o'clock position)",
    "AN5": "Low-speed shaft (LS-SH)",
    "AN6": "Intermediate-speed shaft (IMS-SH)",
    "AN7": "High-speed shaft (HS-SH)",
    "AN8": "High-speed shaft upwind bearing",
    "AN9": "High-speed shaft downwind bearing",
    "AN10": "Planet carrier (downwind)"
}

# Mapping from components to their possible failure modes
# Based on the information from sensors-damage.md
COMPONENT_FAILURE_MODES = {
    "Ring gear (6 o'clock position)": ["Scuffing and polishing", "Fretting corrosion"],
    "Ring gear (12 o'clock position)": ["Scuffing and polishing", "Fretting corrosion"],
    "Low-speed shaft (LS-SH)": ["Fretting corrosion", "Polishing wear"],
    "Intermediate-speed shaft (IMS-SH)": ["Fretting corrosion", "Scuffing", "Assembly damage"],
    "High-speed shaft (HS-SH)": ["Scuffing"],
    "High-speed shaft upwind bearing": ["Assembly damage", "Scuffing", "Dents"],
    "High-speed shaft downwind bearing": ["Overheating", "Assembly damage", "Dents"],
    "Planet carrier (downwind)": ["Fretting corrosion"]
}

# Sensors that typically see anomalies together 
# (based on mechanical coupling in gearbox)
COUPLED_SENSORS = [
    ["AN3", "AN4"],                  # Ring gear positions are coupled
    ["AN6", "AN9"],                  # IMS-SH and HS-SH downwind bearing often fail together
    ["AN8", "AN9"],                  # Both HS-SH bearings often show issues together
    ["AN5", "AN10"],                 # LS-SH and carrier are mechanically linked
    ["AN7", "AN8", "AN9"],           # Complete high-speed section
    ["AN3", "AN4", "AN5", "AN10"]    # Ring gear and planet carrier system
]

# Primary diagnostic rules based on known failure patterns
# Maps combinations of sensor anomalies to the most likely component and failure mode
DIAGNOSTIC_RULES = [
    # Add a specific rule for AN3 to fix the "Unknown" mode issue
    {
        "sensors": ["AN3"],
        "component": "Ring gear (6 o'clock position)",
        "failure_mode": "Scuffing and polishing"
    },
    {
        "sensors": ["AN9"],
        "component": "High-speed shaft downwind bearing",
        "failure_mode": "Overheating"
    },
    {
        "sensors": ["AN8", "AN9"],
        "component": "High-speed shaft bearings",
        "failure_mode": "Assembly damage"
    },
    {
        "sensors": ["AN7", "AN8", "AN9"],
        "component": "High-speed section",
        "failure_mode": "Scuffing"
    },
    {
        "sensors": ["AN6"],
        "component": "Intermediate-speed shaft (IMS-SH)",
        "failure_mode": "Fretting corrosion"
    },
    {
        "sensors": ["AN6", "AN4"],
        "component": "IMS-SH upwind bearing",
        "failure_mode": "Assembly damage and dents"
    },
    {
        "sensors": ["AN3", "AN4"],
        "component": "Ring gear",
        "failure_mode": "Scuffing and polishing"
    },
    {
        "sensors": ["AN10"],
        "component": "Planet carrier upwind bearing",
        "failure_mode": "Fretting corrosion"
    },
    {
        "sensors": ["AN5"],
        "component": "Low-speed shaft",
        "failure_mode": "Fretting corrosion"
    }
]

# Sensor sensitivity adjustments - some sensors require different thresholds
# to accurately detect anomalies (based on empirical testing)
SENSOR_SENSITIVITY = {
    "AN3": 0.85,   # More sensitive (was 1.0)
    "AN4": 0.75,   # Significantly more sensitive (was 0.85)
    "AN5": 0.85,   # More sensitive (was 1.0)
    "AN6": 0.8,    # More sensitive (was 0.9)
    "AN7": 0.8,    # More sensitive (was 0.85)
    "AN8": 0.8,    # More sensitive (was 0.9)
    "AN9": 0.85,   # More sensitive (was 0.9)
    "AN10": 0.85   # More sensitive (was 0.95)
}

# Define correlations between sensor signals and failure modes
# Higher value = stronger correlation with this mode
SENSOR_MODE_CORRELATIONS = {
    "AN3": {
        "Scuffing and polishing": 0.8,
        "Fretting corrosion": 0.6
    },
    "AN4": {
        "Scuffing and polishing": 0.8,
        "Fretting corrosion": 0.6
    },
    "AN5": {
        "Fretting corrosion": 0.9,
        "Polishing wear": 0.7
    },
    "AN6": {
        "Fretting corrosion": 0.7,
        "Scuffing": 0.6,
        "Assembly damage": 0.5
    },
    "AN7": {
        "Scuffing": 0.9
    },
    "AN8": {
        "Assembly damage": 0.8,
        "Scuffing": 0.7,
        "Dents": 0.6
    },
    "AN9": {
        "Overheating": 0.9,
        "Assembly damage": 0.7,
        "Dents": 0.6
    },
    "AN10": {
        "Fretting corrosion": 0.9
    }
}

# Severity thresholds for anomaly scores
SEVERITY_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8
}
