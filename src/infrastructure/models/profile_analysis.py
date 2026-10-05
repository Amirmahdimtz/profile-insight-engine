from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.context.sql_db.psql_dbcontext import Base


class ProfileAnalysisModel(Base):
    __tablename__ = "profile_analysis"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'in_progress', 'completed', 'failed')",
            name="ck_profile_analysis_status",
        ),
        CheckConstraint(
            "((status = 'completed' AND result_payload IS NOT NULL) OR "
            "(status <> 'completed' AND result_payload IS NULL))",
            name="ck_profile_analysis_result_state",
        ),
    )

    id: Mapped[str] = mapped_column("analysis_id", Text, primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    image_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    result_payload: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    creation_datetime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    modification_datetime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
