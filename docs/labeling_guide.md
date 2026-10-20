# Labeling guide — 9 Thai traffic classes

> Status: **frozen** (D-003 `Accepted`, user confirmed 2026-10-20). The order below is the `class_id`
> order in `configs/data/thai_traffic.yaml`. Changing the order or count breaks every checkpoint.
> Example images are not included yet: no Thai footage exists (D-016). Add 2–3 per class during
> phase-3-d (EDA) and phase-3-f, with face/plate-free crops only (`docs/privacy.md`).

## Rules for every box

- One box per visible vehicle, tight around the vehicle body including mirrors; exclude the rider's
  extended limbs and cargo that sticks out far beyond the body.
- Label vehicles that are ≥ 50% visible. Skip vehicles that are smaller than 10×10 px.
- A rider on a motorcycle or bicycle belongs inside the vehicle box (no separate person class).
- Parked vehicles are labelled too; the counting line, not the label, decides what is counted.
- Towed or carried vehicles: label the carrier only.
- If you cannot tell two classes apart, use the rule in "Ambiguous cases" below, never guess differently
  from image to image.

## Classes

| id | Class | Definition | Not this |
|---|---|---|---|
| 0 | `car` | Passenger car with a closed cabin: sedan, hatchback, SUV, taxi, MPV | Pickup (open bed), van (large cargo/passenger body) |
| 1 | `pickup` | Cabin plus open cargo bed, including with a canopy or cap and 4-door double cab | Songthaew (benches/roof built for passengers), small truck with a fixed box body |
| 2 | `van` | Box-shaped passenger or cargo van: Toyota Hiace/Commuter, delivery vans | Car (MPV with normal car roofline), minibus-sized bus |
| 3 | `bus` | Large passenger vehicle for 20+ people: city bus, coach, school bus | Van/minibus, songthaew |
| 4 | `truck` | Rigid or articulated goods vehicle: 6-wheel, 10-wheel, trailer, tanker, dump truck | Pickup, van |
| 5 | `motorcycle` | Two-wheeled motorised vehicle, including scooters and motorcycle taxis, with rider(s) | Bicycle, tuk_tuk (three wheels) |
| 6 | `bicycle` | Pedal-powered two-wheeler, with rider | Motorcycle, e-bikes with a motor and no pedals |
| 7 | `tuk_tuk` | Three-wheeled motorised vehicle: tuk-tuk, samlor, motorcycle with sidecar | Songthaew, motorcycle |
| 8 | `songthaew` | Pickup- or small-truck-based shared taxi with benches in the rear and a roof over the passenger area | Pickup with a plain canopy, bus |

## Ambiguous cases

| Situation | Rule |
|---|---|
| Pickup vs small truck | Open bed and a car-sized cabin → `pickup`. Fixed box or cargo body behind the cabin → `truck` |
| Pickup with a canopy | Still `pickup`, unless benches and open sides show passenger use → `songthaew` |
| Van vs car (MPV) | Sliding rear door and a boxy van roofline → `van`. Car roofline → `car` |
| Minibus vs van vs bus | Hiace/Commuter size → `van`. Rigid body with a bus-size length and a driver separate from the passenger area → `bus` |
| E-scooter / e-bike | Has a motor and a rider seated like a scooter → `motorcycle`. Pedal-assisted bicycle → `bicycle` |
| Tricycle cargo bike (pedal) | `bicycle` |
| Heavily occluded (< 50% visible) | skip |
| Cannot decide after 5 seconds | skip the box and note the image in `docs/label_issues.md` (phase-3-e) |

## Source datasets and the taxonomy

- COCO `truck` mixes pickups and trucks → relabel a sample (flagged in `thai_traffic.yaml`).
- `visaitech_mixed` has only coarse classes (`4-wheeler` mixes car/pickup/van/bus/truck), so it is
  not mapped to `car` (see `configs/data/sources.yaml`).
- Roboflow Thai `Taxi` maps to `car`.
