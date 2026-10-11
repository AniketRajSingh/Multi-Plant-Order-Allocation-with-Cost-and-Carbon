# Data Card — Synthetic Supply Chain Dataset

**Dataset Name:** `mpoa-synthetic`  
**Version:** `1.0`  
**Date:** 2026-10-11  
**Format:** Conforms to HCLTech Hackathon Annex B & Annex D Specifications  

---

## 1. How It Was Made
- **Generation Method:** Algorithmic synthetic generation script using NumPy, Pandas, and seeded random distributions.
- **Seed:** `42` (ensures 100% reproducible generation across runs).
- **Planning Start Date:** `2026-10-12`

---

## 2. What It Contains

The dataset pack consists of 4 primary tables (and 1 drop table):

| Table | File Path | Row Count | Description |
|---|---|:---:|---|
| `orders` | `data_syn/orders.csv` | 150 | Global customer orders (order_id, product, qty, destination, due date) |
| `plants` | `data_syn/plants.csv` | 11 | Manufacturing plant specifications across Plants A, B, and C |
| `transport_duties` | `data_syn/transport_duties.csv` | 36 | 3 plants × 12 international destinations with freight, transit days, tariffs, and carbon |
| `plant_carbon_weekly` | `data_syn/plant_carbon_weekly.csv` | 24 | Weekly grid carbon fluctuations across planning periods |
| `capacity_events` | `data_syn/drop/capacity_events.csv` | 2 | Plant disruption events (Plant B -30% capacity drop) |

### Date Ranges:
- **Planning Start:** `2026-10-12`
- **Order Due Dates:** `2026-10-24` to `2026-12-21`
- **Disruption Window:** `2026-10-15` to `2026-10-29`

---

## 3. Traps and Edge Cases Added on Purpose (Section 4.4)

1. **Cheapest Plant Misses Delivery Date (Order O-077):**
   - Plant A has the lowest landed production cost for Order O-077, but transit takes too long, arriving 3 days past the customer promise.
   - Plant C is slightly higher in landed cost ($18,200 vs $17,400) but arrives on time. The optimizer correctly picks Plant C.
2. **Single-Plant Product (Product P5):**
   - Product P5 can only be produced at Plant C. Assigning P5 to Plant A or B is physically ineligible (Annex A Rule 2).
3. **Import Tariff Flip Trap:**
   - Destination `US` has a 40% import duty on Plant A shipments, flipping Plant A from the cheapest FOB producer to the most expensive landed cost option.
4. **Low-Carbon Small Plant Capacity Trap:**
   - Plant C operates on green solar energy (low carbon), but its weekly capacity is small (2,400 units). Greedy green assignment causes capacity exhaustion.
5. **Mid-Competition Capacity Drop:**
   - Capacity event `E-1` cuts Plant B's capacity by 30%, triggering multi-order re-routing to Plants A and C.

---

## 4. Checks Run (Section 4.5 Validation Results)

Validated using `scripts/validate_data.py --dir data_syn`:
- [PASS] Every order's product can be made in at least one plant.
- [PASS] Every plant-destination lane has a matching transport & duty record (36/36 lanes).
- [PASS] Weekly capacities are positive integers.
- [PASS] Due dates are strictly after planning start (`2026-10-12`).
- [PASS] Carbon intensity values are within expected engineering bounds ($0.04 - 0.72\text{ kg CO}_2\text{e/kWh}$).
- [PASS] Row counts match schema declaration exactly (150 orders, 11 plant rows, 36 lanes, 2 events).

---

## 5. Known Limitations
- Does not model sub-assembly component BOMs or raw material supplier lead times.
- Transit times assume normal weather and port conditions without customs clearance delays.
- Tariffs assume static trade agreements across the single planning window.

---

## 6. Privacy & Synthetic Data Confirmation
- **Confirmation:** We explicitly confirm that **NO real company data, confidential corporate numbers, or personally identifiable information (PII)** is present anywhere in this dataset. All order IDs, customer destinations, and plant parameters are 100% synthetically generated.
