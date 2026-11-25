from parking.perception.occupancy import OccupancyFilter, assign_occupancy, overlap_ratio


def test_overlap_full_cover():
    # unit square slot, vehicle covers most of it
    space = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ratio = overlap_ratio(space, (10, 10, 90, 90), width=100, height=100)
    assert ratio > 0.5


def test_assign_one_car_one_slot():
    spaces = [
        {"space_id": "S1", "points": [[0.0, 0.0], [0.4, 0.0], [0.4, 0.5], [0.0, 0.5]]},
        {"space_id": "S2", "points": [[0.6, 0.0], [1.0, 0.0], [1.0, 0.5], [0.6, 0.5]]},
    ]
    vehicles = [{"xyxy": (5, 5, 35, 45), "conf": 0.9, "label": "car"}]
    states = assign_occupancy(spaces, vehicles, width=100, height=100, ratio_threshold=0.1)
    by_id = {s["space_id"]: s["state"] for s in states}
    assert by_id["S1"] == "occupied"
    assert by_id["S2"] == "vacant"


def test_occupancy_filter_needs_hits():
    filt = OccupancyFilter(hits_needed=2)
    first = filt.update([{"space_id": "S1", "state": "occupied", "confidence": 0.8}])
    assert first[0]["state"] == "occupied"  # unknown → first raw accepted until confirmed
    # establish vacant
    filt2 = OccupancyFilter(hits_needed=2)
    filt2.update([{"space_id": "S1", "state": "vacant", "confidence": 1.0}])
    filt2.update([{"space_id": "S1", "state": "vacant", "confidence": 1.0}])
    one = filt2.update([{"space_id": "S1", "state": "occupied", "confidence": 0.9}])
    assert one[0]["state"] == "vacant"
    two = filt2.update([{"space_id": "S1", "state": "occupied", "confidence": 0.9}])
    assert two[0]["state"] == "occupied"
