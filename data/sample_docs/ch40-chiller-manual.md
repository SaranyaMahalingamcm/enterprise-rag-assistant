---
doc_id: ch40-manual
title: CH-40 Recirculating Chiller Manual
doc_type: manual
equipment: CH-40
revision: D
---

# CH-40 Recirculating Chiller Manual

> Synthetic document for a fictional manufacturer (Acme Fab Tools).

## Purpose

The CH-40 recirculating chiller supplies temperature-controlled coolant to the LC-500 laser tube and
the VC-90 magnetron cathode. Cooling capacity is 4 kW at 20 degrees Celsius.

## Coolant Specification

Fill the reservoir with a mixture of 90 percent deionized water and 10 percent AFT-COOL corrosion
inhibitor. Never use tap water, which causes scale in the laser tube. Replace the coolant every
6 months or when conductivity exceeds 20 microsiemens per centimeter.

## Setpoints

- Default setpoint: 20 degrees Celsius.
- Allowed range for the LC-500: 18 to 22 degrees Celsius.
- High-temperature alarm threshold: 26 degrees Celsius.

## Filter Service

Clean the air-side condenser filter every 2 weeks with low-pressure compressed air. A clogged filter
is the most common cause of high-temperature alarms in summer.

## Chiller Alarm Codes

- **C01 Low reservoir level**: top up with premixed coolant.
- **C02 High coolant temperature**: check the condenser filter and ambient temperature (maximum 35 degrees Celsius).
- **C05 Flow sensor fault**: inspect the flow sensor connector; if the fault persists, raise a Tier 2 service request.
