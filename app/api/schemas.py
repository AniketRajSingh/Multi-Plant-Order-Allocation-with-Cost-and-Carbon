"""Pydantic v2 schemas for the Section 6 API contract and the Annex B data files.

Two groups live here:
- API models: request/response bodies for /analyze, /ingest, /health and /trace.
- Row models: one per Annex B table. The ingest path validates every CSV row
  against these, so a bad row is rejected with a reason instead of silently
  corrupting the store. Extra columns are allowed (Annex B says so) and kept.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AnswerType = Literal[
    "allocation",
    "scenario",
    "why_order",
    "brief",
    "infeasible",
    "clarification_needed",
    "refused",
]

TableName = Literal["orders", "plants", "transport_duties", "capacity_events"]
IngestMode = Literal["replace", "append", "upsert"]


# ============================================================================
# ANNEX B ROW SCHEMAS (used by the ingest path, one model per table)
# ============================================================================

class _Row(BaseModel):
    """Base row: strips whitespace, keeps extra columns."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)


class OrderRow(_Row):
    order_id: str = Field(min_length=1)
    product: str = Field(min_length=1)
    qty: int = Field(gt=0, description="Units ordered")
    destination: str = Field(min_length=1)
    due: date


class PlantRow(_Row):
    plant: str = Field(min_length=1)
    product: str = Field(min_length=1)
    weekly_capacity: int = Field(gt=0, description="Units per week")
    unit_cost: float = Field(ge=0, description="Production cost per unit")
    energy_kwh_per_unit: float = Field(ge=0, description="kWh per unit")
    carbon_kg_per_kwh: float = Field(ge=0, description="kg CO2e per kWh")

    @field_validator("plant")
    @classmethod
    def _upper_plant(cls, v: str) -> str:
        return v.upper()


class TransportDutyRow(_Row):
    plant: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    transport_cost_per_unit: float = Field(ge=0)
    transit_days: int = Field(ge=0)
    duty_pct: float = Field(ge=0, le=100, description="Import duty, percent of value")
    carbon_kg_per_unit: float = Field(ge=0, description="Transport kg CO2e per unit")

    @field_validator("plant")
    @classmethod
    def _upper_plant(cls, v: str) -> str:
        return v.upper()


class CapacityEventRow(_Row):
    event_id: str = Field(min_length=1)
    plant: str = Field(min_length=1)
    from_date: date
    to_date: date
    capacity_loss_pct: float = Field(ge=0, le=100, description="Percent of capacity lost")

    @field_validator("plant")
    @classmethod
    def _upper_plant(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _dates_ordered(self) -> "CapacityEventRow":
        if self.to_date < self.from_date:
            raise ValueError("to_date is before from_date")
        return self


ROW_MODELS: Dict[str, type[_Row]] = {
    "orders": OrderRow,
    "plants": PlantRow,
    "transport_duties": TransportDutyRow,
    "capacity_events": CapacityEventRow,
}


# ============================================================================
# POST /analyze
# ============================================================================

class Weights(BaseModel):
    """Objective weights. Omitted fields fall back to the defaults (cost 0.7, carbon 0.3)."""

    model_config = ConfigDict(extra="forbid")

    cost: Optional[float] = Field(default=None, ge=0, le=1)
    carbon: Optional[float] = Field(default=None, ge=0, le=1)
    lateness: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _not_all_zero(self) -> "Weights":
        values = [v for v in (self.cost, self.carbon, self.lateness) if v is not None]
        if values and sum(values) == 0:
            raise ValueError("at least one weight must be greater than zero")
        return self


class Scenario(BaseModel):
    """What-if input, e.g. a capacity change. Extra keys pass through to the orchestrator."""

    model_config = ConfigDict(extra="allow")

    event_id: Optional[str] = None
    plant: Optional[str] = None
    capacity_loss_pct: Optional[float] = Field(default=None, ge=0, le=100)
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    description: Optional[str] = None

    @field_validator("plant")
    @classmethod
    def _upper_plant(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if v else v


class AnalyzeRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4000)
    weights: Optional[Weights] = None
    scenario: Optional[Scenario] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"request": "Allocate this week's orders."},
                {
                    "request": "Plant B loses 30% capacity. Re-plan.",
                    "weights": {"cost": 0.7, "carbon": 0.3},
                    "scenario": {"plant": "B", "capacity_loss_pct": 30},
                },
            ]
        }
    )


class AllocationItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    order_id: str
    plant: Optional[str] = None
    cost: Optional[float] = None
    carbon_kg: Optional[float] = None
    arrival: Optional[str] = None
    on_time: Optional[bool] = None


class KPIs(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_cost: Optional[float] = None
    total_carbon_t: Optional[float] = None
    late_orders: Optional[int] = None
    # null when a plant's effective capacity is 0 (e.g. a 100% capacity loss)
    plant_utilisation: Dict[str, Optional[float]] = Field(default_factory=dict)


class TradeoffPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    carbon_cut_pct: Optional[float] = None
    cost: Optional[float] = None
    late_orders: Optional[int] = None


class ValidatorResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    capacity_ok: Optional[bool] = None
    eligibility_ok: Optional[bool] = None
    delivery_ok: Optional[bool] = None
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """Section 6.1: every recommendation carries these six fields."""

    action: str
    rationale: str
    evidence_refs: List[str] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    estimated_impact: str
    requires_approval: bool


class AnalyzeResponse(BaseModel):
    """Section 6.1 response. The first eleven fields are the contract; the rest are extras."""

    run_id: str
    answer_type: AnswerType
    summary: str
    allocation: List[AllocationItem] = Field(default_factory=list)
    kpis: Optional[KPIs] = None
    tradeoff: List[TradeoffPoint] = Field(default_factory=list)
    weights_used: Dict[str, float] = Field(default_factory=dict)
    validator: Optional[ValidatorResult] = None
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)

    brief: Optional[str] = None
    scenario_comparison: Optional[Dict[str, Any]] = None
    unallocated_orders: List[Dict[str, Any]] = Field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    latency_ms: float = 0.0


# ============================================================================
# POST /ingest
# ============================================================================

class IngestMetadata(BaseModel):
    """The JSON part of the multipart upload."""

    model_config = ConfigDict(extra="ignore")

    table: TableName
    mode: IngestMode = "replace"


class RejectedRow(BaseModel):
    row_number: int = Field(description="1-based data row number in the CSV (header excluded)")
    reason: str


class IngestResponse(BaseModel):
    table: str
    rows_loaded: int
    rows_rejected: int
    status: Literal["loaded", "partial", "failed"]
    mode: IngestMode = "replace"
    total_rows_in_table: int = 0
    errors: List[str] = Field(default_factory=list)
    rejected: List[RejectedRow] = Field(default_factory=list, description="First 50 rejected rows")


# ============================================================================
# GET /health
# ============================================================================

class ComponentHealth(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["ok", "degraded", "down"]
    detail: Optional[str] = None


class DataStoreHealth(ComponentHealth):
    path: str
    tables: Dict[str, int] = Field(default_factory=dict)


class LLMHealth(ComponentHealth):
    provider: str
    model: str
    mock: bool


class SolverHealth(ComponentHealth):
    name: str
    version: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    api: ComponentHealth
    data_store: DataStoreHealth
    solver: SolverHealth
    llm: LLMHealth


# ============================================================================
# GET /trace/{run_id}  (Annex C)
# ============================================================================

class TraceStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent: str
    tool: str
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None


class TraceResponse(BaseModel):
    run_id: str
    request: str
    steps: List[TraceStep]
    llm_calls: int = 0
    tokens: int = 0


class ErrorResponse(BaseModel):
    detail: str
