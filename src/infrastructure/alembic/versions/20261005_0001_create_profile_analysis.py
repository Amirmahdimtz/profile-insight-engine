"""create profile analysis persistence

Revision ID: 20261005_0001
Revises:
Create Date: 2026-10-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20261005_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "profile_analysis",
        sa.Column("analysis_id", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("image_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "result_payload",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "creation_datetime",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "modification_datetime",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'in_progress', 'completed', 'failed')",
            name="ck_profile_analysis_status",
        ),
        sa.CheckConstraint(
            "((status = 'completed' AND result_payload IS NOT NULL) OR "
            "(status <> 'completed' AND result_payload IS NULL))",
            name="ck_profile_analysis_result_state",
        ),
        sa.PrimaryKeyConstraint("analysis_id"),
    )
    op.create_index(
        "ix_profile_analysis_status",
        "profile_analysis",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_profile_analysis_status", table_name="profile_analysis")
    op.drop_table("profile_analysis")
