---
doc_id: sop-014-robot-calibration
title: SOP-014 RA-6 Robot Arm TCP Calibration and Optical Alignment
doc_type: sop
equipment: RA-6
revision: 5
---

# SOP-014 RA-6 Robot Arm TCP Calibration and Optical Alignment

> Synthetic document for a fictional manufacturer (Acme Fab Tools).

## When to Calibrate

Calibrate the RA-6 tool center point (TCP) quarterly, after any end-effector change, after a
collision, or whenever alarm R22 appears.

## Four-Point TCP Calibration

1. Mount the calibration pin (part AFT-RA-CP6) in the end effector.
2. Jog the robot so the pin tip touches the fixed reference cone from four clearly different orientations.
3. Record each pose with the teach pendant under menu Setup > Tool > 4-Point.
4. Accept the result only if the reported TCP error is 0.2 mm or less. Otherwise repeat with
   orientations that differ by at least 30 degrees.

## LC-500 Mirror Alignment (Target Card Method)

Place a target card at each mirror in turn and fire a single low-power test pulse (5 percent power,
10 ms). Adjust the mirror screws until the burn mark is centered on the card at both the near and
far ends of the gantry travel. Always wear the laser safety glasses rated for 10.6 micrometers.
