---
doc_id: cm220-maintenance
title: CM-220 CNC Mill Maintenance Guide
doc_type: manual
equipment: CM-220
revision: B
---

# CM-220 CNC Mill Maintenance Guide

> Synthetic document for a fictional manufacturer (Acme Fab Tools).

## Spindle Lubrication

The CM-220 spindle uses a sealed grease-packed bearing cartridge and requires no routine greasing.
However, the spindle taper must be wiped clean and lightly oiled with ISO VG 10 spindle oil at the
start of every shift to prevent fretting corrosion.

## Way Oil System

The automatic way-oil pump dispenses 3 cc every 30 minutes of axis motion. Use ISO VG 68 way oil only.
Check the reservoir sight glass daily and refill before the level falls below the MIN mark. An empty
reservoir triggers alarm E220 and halts axis motion.

## Backlash Check

Measure backlash on the X, Y and Z axes every 500 operating hours using a dial indicator with
0.001 mm resolution. Acceptable backlash is 0.005 mm or less per axis. If an axis exceeds 0.010 mm,
enter the value in controller parameter P-1851 as a temporary compensation and raise a Tier 2
service request to inspect the ball screw.

## Coolant Concentration

Maintain water-soluble coolant concentration at 6 to 8 percent, measured with a refractometer at the
start of each week. Low concentration leads to rust on the table; high concentration causes foaming
and skin irritation.

## Tool Changer Maintenance

Clean the 20-pocket carousel and pull studs monthly. Replace pull studs every 2,000 tool changes
(part AFT-CM-PS40).
