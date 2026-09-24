"""Open a romanmock parent catalog and compute the diagnostics of the plan."""

import sys

import numpy as np

from romanmock.diagnostics import completeness, contamination_fraction, empirical_confusion, outcome_summary
from romanmock.pipeline import RomanMockPipeline

pipe = RomanMockPipeline.from_yaml(sys.argv[1])
parent = pipe.open_parent()
cols = ["redshift_assigned", "redshift_rsd", "id_outcome", "n_det", "line_true", "line_assigned"]
data = parent.select(*cols).get_data("numpy", wrap_single=True)

print(outcome_summary(data))
print(np.round(empirical_confusion(data, pipe.line_set), 4))

z_edges = np.linspace(1.0, 2.0, 11)
cont = contamination_fraction(data, pipe.line_set, z_edges, assigned_line="Ha")
for i, name in enumerate(cont["lines"]):
    print(f"{name:>10s} -> Ha interloper fraction per bin:", np.round(cont["fraction"][i], 4))
comp = completeness(data, pipe.line_set, z_edges, line="Ha")
print("Ha completeness per true-z bin:", np.round(comp["completeness"], 4))
