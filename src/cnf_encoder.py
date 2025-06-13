# src/cnf_encoder.py
from pysat.formula import CNF

class CNFEncoder:
    def __init__(self, alphabet="abcdefghijklmnopqrstuvwxyz"):
        self.alphabet = alphabet
        self.L = len(alphabet)

    def var(self, i, l):
        """1-based DIMACS variable ID for position i, label index l."""
        return i*self.L + l + 1

    def encode_word(self, n, invalid_bigrams=None):
        cnf = CNF()
        # 4.a Exactly-one per position
        for i in range(n):
            # At least one
            cnf.append([self.var(i,l) for l in range(self.L)])
            # At most one  (pairwise encoding)
            for l1 in range(self.L):
                for l2 in range(l1+1, self.L):
                    cnf.append([-self.var(i,l1), -self.var(i,l2)])
        # 4.b Bigram constraints (optional)
        if invalid_bigrams:
            for i in range(n-1):
                for (l1,l2) in invalid_bigrams:
                    cnf.append([-self.var(i,l1), -self.var(i+1,l2)])
        return cnf
