from __future__ import annotations

from typing import Optional, get_args, get_origin

from pydantic import BaseModel


# Map Python types to SQL column types.
_TYPE_MAP: dict[type, str] = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "INTEGER",
    bytes: "BLOB",
}


def _sql_type(annotation: object) -> tuple[str, bool]:
    """Return the SQL type name and whether the column is nullable.

    Args:
        annotation: A Python type annotation (possibly Optional[T]).

    Returns:
        Tuple of (sql_type_string, is_nullable).
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is not None and args:
        # Optional[T] is Union[T, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            sql_t, _ = _sql_type(non_none[0])
            return sql_t, True
        return "TEXT", True

    return _TYPE_MAP.get(annotation, "TEXT"), False  # type: ignore[arg-type]


def derive_ddl(schema_cls: type[BaseModel], table_name: str) -> str:
    """Derive a CREATE TABLE statement from a Pydantic model class.

    Primary keys are identified by the string "Primary key" in the field description.
    Foreign keys are identified by "Foreign key referencing <table>" in the description.

    Args:
        schema_cls: Pydantic model class whose fields describe the table structure.
        table_name: Name to use in the CREATE TABLE statement.

    Returns:
        SQL DDL string (CREATE TABLE IF NOT EXISTS ...).
    """
    # Fields to skip — they are schema metadata, not table columns.
    _META_FIELDS = {"schema_name", "schema_version"}

    column_lines: list[str] = []
    fk_lines: list[str] = []

    for field_name, field_info in schema_cls.model_fields.items():
        if field_name in _META_FIELDS:
            continue

        description: str = field_info.description or ""
        sql_type, nullable = _sql_type(field_info.annotation)

        parts: list[str] = [field_name, sql_type]

        is_pk = "primary key" in description.lower()
        is_fk = "foreign key referencing" in description.lower()

        if is_pk:
            parts.append("PRIMARY KEY")
        elif not nullable:
            parts.append("NOT NULL")

        column_lines.append("    " + " ".join(parts))

        if is_fk:
            lower = description.lower()
            ref_start = lower.index("foreign key referencing") + len("foreign key referencing")
            ref_text = description[ref_start:].strip().split()[0].rstrip(".,;")
            if "(" in ref_text:
                ref_table, ref_col = ref_text.rstrip(")").split("(")
            else:
                ref_table = ref_text
                ref_col = field_name
            fk_lines.append(
                f"    FOREIGN KEY ({field_name}) REFERENCES {ref_table}({ref_col})"
            )

    all_lines = column_lines + fk_lines
    body = ",\n".join(all_lines)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n{body}\n);"
