# src/sat_infer.py
from pysat.solvers import Minisat22

class SATInfer:
    def __init__(self, encoder):
        self.encoder = encoder

    def decode(self, scores):
        """scores: shape (n, L) real-valued potentials.  
        We add soft scores as unit clauses via weight -> assumption trick."""
        n, L = scores.shape
        cnf = self.encoder.encode_word(n)
        # turn argmax into SAT: pick highest score after satisfying hard clauses
        # simplest: brute-force all labels by descending score until SAT.
        order = [(i,l,scores[i,l]) for i in range(n) for l in range(L)]
        order.sort(key=lambda t: -t[2])
        with Minisat22(bootstrap_with=cnf.clauses) as m:
            assignment = [None]*n
            for i,l,_ in order:
                if assignment[i] is None:
                    m.add_clause([ self.encoder.var(i,l) ])
                    if m.solve():
                        model = m.get_model()
                        assignment = [None]*n
                        for i2 in range(n):
                            for l2 in range(L):
                                if model[self.encoder.var(i2,l2)-1] > 0:
                                    assignment[i2] = l2
                                    break
                    else:
                        m.add_clause([-self.encoder.var(i,l)])  # backtrack
            return assignment
