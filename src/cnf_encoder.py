# cnf_encoder.py
import itertools
import numpy as np
from pysat.formula import WCNF


class CNFEncoder:
    """
    Ultra-light CNF / Max-SAT encoder.

    • Variables are positive integers 1, 2, 3, … (DIMACS style)
    • Soft clauses get a finite weight
    • Hard clauses get weight = topw (PySAT convention)
    """

    def __init__(self):
        self.reset()
    
    # ---------- low-level helpers ----------
    def reset(self):
        self.next_var = 1       # DIMACS vars start at 1
        self.wcnf = WCNF()

    def new_var(self):
        v = self.next_var
        self.next_var += 1
        return v
    
    def _canon(self, lits):
        if isinstance(lits, (int, np.integer)):
            return [int(lits)]
        return [int(l) for l in lits]

    def _append(self, lits, weight):
        self.wcnf.append(self._canon(lits), weight=weight)

    # ---------- clause helpers ----------
    def add_soft(self, clause, weight):
        self._append(clause, weight=weight)

    def add_hard(self, clause):
        self._append(clause, weight=self.wcnf.topw) 

    # ---------- composite constraints ----------
    def exactly_one(self, vars_):
        # at least one
        self.add_hard(list(vars_))
        # at most one  (pairwise)
        for i, j in itertools.combinations(vars_, 2):
            self.add_hard([-i, -j])

    # ---------- final CNF ----------
    def to_wcnf(self):
        return self.wcnf
