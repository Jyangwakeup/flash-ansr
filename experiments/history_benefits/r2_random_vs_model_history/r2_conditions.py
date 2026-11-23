"""The six final-evaluation conditions of round 2 (moved unchanged from evaluation.py)."""

#: The six final-evaluation conditions (流程 §7.2): name -> (group, history source or None).
CONDITIONS = {
    "1_A_none": ("A", None),
    "2_C_model": ("C", "model"),
    "3_B_random": ("B", "random"),
    "4_C_none": ("C", None),
    "5_B_none": ("B", None),
    "6_B_model": ("B", "model"),
}
