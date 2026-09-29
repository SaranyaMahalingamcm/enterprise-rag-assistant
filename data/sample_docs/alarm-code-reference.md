---
doc_id: alarm-reference
title: Controller Alarm Code Reference
doc_type: reference
equipment: all
revision: 12
---

# Controller Alarm Code Reference

> Synthetic document for a fictional manufacturer (Acme Fab Tools).

## How to Read Alarm Codes

Alarm codes are shown on the controller display as a letter prefix and a number. The prefix
identifies the subsystem: E for machine controller alarms, C for chiller alarms and R for robot alarms.
Alarms numbered 1xx are LC-500 alarms, 2xx are CM-220 alarms and 3xx are VC-90 alarms.

## Selected Machine Alarms

- **E101** LC-500 door interlock open. Operator can clear.
- **E104** LC-500 coolant flow low. Operator can clear after checking the chiller.
- **E110** LC-500 laser tube over-temperature. Requires 15 minute cool-down.
- **E220** CM-220 way oil reservoir empty. Refill ISO VG 68 way oil, then reset.
- **E231** CM-220 spindle drive overload. Reduce feed rate; if repeated, Tier 2 service request.
- **E305** VC-90 turbo pump failed to reach speed. Check roughing pressure and turbo controller cable.
- **E312** VC-90 base pressure not reached within 40 minutes. Perform leak check per TS-003.

## Robot Alarms

- **R17** RA-6 joint position following error. Check for collision or payload overload.
- **R22** RA-6 TCP calibration out of tolerance. Recalibrate per SOP-014.

## Escalation

Any alarm that repeats three times in one shift must be logged and escalated as a Tier 2 service request.
