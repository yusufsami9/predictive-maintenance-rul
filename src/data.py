"""Column layout shared by every C-MAPSS subset (FD001-FD004)."""

COLUMN_NAMES = ["unit", "cycle", "op1", "op2", "op3"] + [f"sensor{i}" for i in range(1, 22)]
