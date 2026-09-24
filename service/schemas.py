"""Pydantic request/response models: validation at the API boundary."""
from pydantic import BaseModel, Field, field_validator

from bandit import ALGORITHMS


class BanditCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80, examples=["homepage-banner"])
    algorithm: str = Field(examples=["Thompson Sampling"])
    n_arms: int = Field(ge=2, le=50, description="Number of ads (arms) to choose between")
    epsilon: float = Field(0.1, ge=0, le=1, description="Epsilon-Greedy / Decaying Epsilon")
    decay_scale: float = Field(50.0, gt=0, description="Decaying Epsilon only")
    ucb_c: float = Field(2.0, ge=0, description="UCB1 exploration constant")

    @field_validator("algorithm")
    @classmethod
    def known_algorithm(cls, value):
        if value not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        return value


class BanditOut(BaseModel):
    id: int
    name: str
    algorithm: str
    n_arms: int
    params: dict
    created_at: str


class SelectOut(BaseModel):
    impression_id: int
    arm: int
    seq: int = Field(description="Position of this impression within the bandit (1-based)")


class ClickOut(BaseModel):
    impression_id: int
    clicked: bool
    duplicate: bool = Field(description="True if this impression already had a click")


class ArmStats(BaseModel):
    arm: int
    impressions: int
    clicks: int
    ctr: float | None


class StatsOut(BaseModel):
    bandit: BanditOut
    impressions: int
    clicks: int
    ctr: float | None
    best_arm_estimate: int | None
    arms: list[ArmStats]


class TrendPoint(BaseModel):
    block: int
    impressions: int
    clicks: int
    ctr: float
    cumulative_ctr: float


class SimulateIn(BaseModel):
    probs: list[float] = Field(min_length=2, max_length=10, examples=[[0.3, 0.5, 0.7]])
    steps: int = Field(2000, ge=100, le=10000)
    n_runs: int = Field(50, ge=1, le=200)
    seed: int | None = Field(42, ge=0)
    shift: bool = Field(False, description="Reverse the click rates halfway through")
    algorithms: list[str] = Field(default_factory=lambda: list(ALGORITHMS), min_length=1)

    @field_validator("probs")
    @classmethod
    def probabilities(cls, value):
        if any(p < 0 or p > 1 for p in value):
            raise ValueError("each probability must be between 0 and 1")
        return value

    @field_validator("algorithms")
    @classmethod
    def known_algorithms(cls, value):
        unknown = [a for a in value if a not in ALGORITHMS]
        if unknown:
            raise ValueError(f"unknown algorithms {unknown}; choose from {ALGORITHMS}")
        return value


class SimulateOut(BaseModel):
    settings: dict
    summary: list[dict]
