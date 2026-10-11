"""Small synthetic data pack for tests. Planning start 2026-10-12.

Traps built in on purpose (Section 4.4):
- O-001, O-007: cheapest plant (A, 9 days to DE) arrives late.
- O-003: product P2 only Plant A can make.
- O-004: 40% import duty makes Plant A the expensive choice for US.
- Plant C: lowest carbon but only 300 units/week.
- E-1: Plant B loses 30% capacity during the planning week.
- Losing Plant A entirely makes O-003 impossible.
"""

from app.db.database import Database

PLANNING_START = "2026-10-12"

PLANTS = """plant,product,weekly_capacity,unit_cost,energy_kwh_per_unit,carbon_kg_per_kwh
A,P1,1000,10.0,2.0,0.70
B,P1,800,12.0,2.0,0.40
C,P1,300,14.0,2.0,0.05
A,P2,1000,20.0,3.0,0.70
A,P3,1000,15.0,2.0,0.70
B,P3,800,16.0,2.0,0.40
"""

TRANSPORT = """plant,destination,transport_cost_per_unit,transit_days,duty_pct,carbon_kg_per_unit
A,DE,1.0,9,0,0.5
B,DE,1.5,4,0,0.4
C,DE,2.0,2,0,0.3
A,FR,1.0,5,0,0.4
B,FR,1.2,3,0,0.3
C,FR,2.2,2,0,0.3
A,US,2.0,6,40,0.8
B,US,3.0,7,0,0.9
C,US,4.0,5,0,0.7
"""

ORDERS = """order_id,product,qty,destination,due
O-001,P1,300,DE,2026-10-18
O-002,P1,200,DE,2026-10-25
O-003,P2,100,FR,2026-10-25
O-004,P1,150,US,2026-10-25
O-005,P3,250,FR,2026-10-20
O-006,P1,100,FR,2026-10-30
O-007,P1,200,DE,2026-10-16
O-008,P3,150,DE,2026-10-30
"""

CAPACITY_EVENTS = """event_id,plant,from_date,to_date,capacity_loss_pct
E-1,B,2026-10-12,2026-10-25,30
"""


def demo_db() -> Database:
    db = Database(":memory:")
    db.init_schema()
    for table, text in [("plants", PLANTS), ("transport_duties", TRANSPORT), ("orders", ORDERS),
                        ("capacity_events", CAPACITY_EVENTS)]:
        res = db.ingest_csv(table, text.encode())
        assert res.status == "loaded", res.to_dict()
    return db
