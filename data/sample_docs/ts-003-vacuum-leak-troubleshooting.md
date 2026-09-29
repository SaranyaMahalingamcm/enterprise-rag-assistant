---
doc_id: ts-003-vacuum-leaks
title: TS-003 Troubleshooting Vacuum Leaks on the VC-90
doc_type: troubleshooting
equipment: VC-90
revision: 2
---

# TS-003 Troubleshooting Vacuum Leaks on the VC-90

> Synthetic document for a fictional manufacturer (Acme Fab Tools).

## Symptoms

- Pump-down to base pressure takes longer than 25 minutes.
- Alarm E312 (base pressure not reached within 40 minutes).
- Coating films look hazy or discolored, which suggests water vapor or air in the chamber.

## Rate-of-Rise Test

Pump the chamber to base pressure, close the high-vacuum valve, and record pressure every minute for
10 minutes. A rise of more than 1 x 10^-4 mbar per minute indicates a real leak. A rise that slows
over time usually indicates outgassing from a dirty chamber rather than a leak.

## Locating the Leak

1. Inspect and clean the lid O-ring first; it causes most leaks.
2. Check the viewport and feedthrough seals for scratches.
3. If a helium leak detector is available, spray helium lightly around each seal starting from the top
   of the chamber and working downward.
4. Tighten KF clamps by hand only; overtightening deforms the centering ring.

## Outgassing

If the rate-of-rise test indicates outgassing, bake the chamber at 60 degrees Celsius for 2 hours
with the roughing pump running, and clean internal shields with IPA per SOP-009.
