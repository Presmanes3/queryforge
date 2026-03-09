from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OrderSchema(BaseModel):
    """Defines the structure of an orders table for DDL derivation and dataset generation."""

    schema_name: str = Field(default="orders", description="Logical name of this schema.")
    schema_version: str = Field(default="v1", description="Semver-style version string.")

    # Table columns
    order_id: int = Field(description="Primary key, auto-incremented integer.")
    customer_id: int = Field(description="Foreign key referencing customers(customer_id).")
    amount: float = Field(description="Total order amount in USD.")
    status: str = Field(
        description="Order lifecycle status: pending, shipped, delivered, cancelled."
    )
    created_at: str = Field(description="ISO 8601 timestamp of order creation.")
    notes: Optional[str] = Field(
        default=None,
        description="Optional free-text notes attached to the order."
    )
