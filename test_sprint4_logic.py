"""
test_sprint4_logic.py
Standalone validation runner for Sprint 4 architectural logic.
"""

from vision.object_detector import DetectedObject, BoundingBox
from vision.detection_manager import DetectionManager
from navigation.navigator import Navigator
from navigation.navigation_types import NavigationAction

def run_scenario(name: str, raw_objects: list):
    print(f"\n{"="*10} Testing Scenario: {name} {"="*10}")
    
    manager = DetectionManager()
    navigator = Navigator()
    
    # Process mock raw inputs using a simulated standard HD width (1280px)
    enriched = manager.process(raw_objects, frame_width=1280)
    decision = navigator.decide(enriched)
    
    print(f"Enriched Detections Count: {len(enriched)}")
    for det in enriched:
        print(f" -> [{det.region}] {det.label} (Priority: {det.priority})")
        
    print(f"SYSTEM DECISION: {decision.action.name}")
    print(f"REASON: {decision.reason}")

# --- DEFINE TEST DATA METRICS ---
# A dummy bounding box for our dataclasts
dummy_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)

# Scenario A: Empty Area
run_scenario("Empty Environment", [])

# Scenario B: Critical Object Blocking Middle
# Center x coordinate at 600px sits cleanly in the middle third of 1280px
run_scenario("Person Center Block", [
    DetectedObject(label="person", confidence=0.95, bbox=BoundingBox(x1=500, y1=100, x2=700, y2=600))
])

# Scenario C: Low Priority Center item (Should NOT command a stop)
run_scenario("Harmless Bottle Center", [
    DetectedObject(label="bottle", confidence=0.88, bbox=BoundingBox(x1=550, y1=400, x2=650, y2=500))
])

# Scenario D: Left Side Obstructed by High-priority target
# Center x coordinate at 200px sits cleanly in the left third
run_scenario("Chair Blocking Left Side", [
    DetectedObject(label="chair", confidence=0.91, bbox=BoundingBox(x1=100, y1=100, x2=300, y2=400))
])

# Scenario E: The Cascade Check (High-priority Left vs Unimportant Center)
# System should handle the Center Bottle first by slowing down, ignoring the left hazard for now!
run_scenario("Cascade Isolation Check (Center Bottle + Left Person)", [
    DetectedObject(label="person", confidence=0.92, bbox=BoundingBox(x1=100, y1=100, x2=300, y2=400)), # LEFT
    DetectedObject(label="bottle", confidence=0.85, bbox=BoundingBox(x1=550, y1=400, x2=650, y2=500))  # CENTER
])
