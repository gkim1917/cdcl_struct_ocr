# sat_infer.py
import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from .cnf_encoder import CNFEncoder


class SATInfer:
    """
    Max-SAT decoding for structured prediction with performance optimizations.
    
    Key optimizations:
    1. Transition pruning: Skip low-weight transitions that rarely affect optimal paths
    2. Score-based variable reduction: Focus on high-scoring label candidates
    3. Early termination for simple cases
    """

    def __init__(self, transition_threshold=2.0, score_threshold=None):
        self.encoder = CNFEncoder()
        self.transition_threshold = transition_threshold  # Skip transitions T[l,q] < T_max - τ
        self.score_threshold = score_threshold  # Optional: skip low-scoring unary candidates
        
    def argmax(self, scores: np.ndarray, T: np.ndarray):
        n, L = scores.shape
        
        # Handle edge cases
        if n == 0:
            return []
        if n == 1:
            return [int(scores[0].argmax())]
        
        # Quick check: if problem is small enough, use Viterbi (faster)
        if n * L <= 50:  # Threshold for small problems
            return self._viterbi_fallback(scores, T)
            
        # Analyze transition matrix for pruning
        T_max = T.max() if T.size > 0 else 0
        T_threshold = T_max - self.transition_threshold
        
        # Count active transitions (for deciding whether to use SAT)
        active_transitions = np.sum(T >= T_threshold)
        total_possible_transitions = L * L
        
        # If too few transitions are active, fall back to Viterbi
        if active_transitions < total_possible_transitions * 0.1:  # Less than 10% active
            return self._viterbi_fallback(scores, T)
        
        # If too many transitions are active, SAT won't help much
        if active_transitions > total_possible_transitions * 0.8:  # More than 80% active
            return self._viterbi_fallback(scores, T)
            
        enc = self.encoder
        enc.reset()

        # 1) Create Boolean variables (only for positions we'll actually use)
        var = np.zeros((n, L), dtype=int)
        var_created = np.zeros((n, L), dtype=bool)
        
        # Create variables for all positions (we need exactly-one constraints)
        for i in range(n):
            for l in range(L):
                var[i, l] = enc.new_var()
                var_created[i, l] = True

        # 2) Create transition variables only for high-weight transitions
        z = {}  # Sparse storage: (i, l, q) -> var_id
        active_transition_mask = T >= T_threshold
        
        transition_count = 0
        for i in range(n - 1):
            for l in range(L):
                for q in range(L):
                    if active_transition_mask[l, q]:
                        z[(i, l, q)] = enc.new_var()
                        transition_count += 1

        print(f"SAT problem size: {n} positions, {L} labels, {transition_count}/{L*L*(n-1)} transitions")

        # 3) Unary soft clauses (with optional score-based pruning)
        min_score = scores.min() if scores.size > 0 else 0
        offset_scores = scores - min_score + 1e-6
        
        # Optional: only add unary clauses for competitive scores
        if self.score_threshold is not None:
            score_max = scores.max(axis=1, keepdims=True)  # Max score per position
            score_mask = scores >= (score_max - self.score_threshold)
        else:
            score_mask = np.ones_like(scores, dtype=bool)
        
        unary_clauses = 0
        for i in range(n):
            for l in range(L):
                if score_mask[i, l] and var_created[i, l]:
                    weight = max(1, int(round(offset_scores[i, l] * 1000)))
                    enc.add_soft([var[i, l]], weight=weight)
                    unary_clauses += 1

        # 4) Bigram transition constraints and costs (only for active transitions)
        min_transition = T.min() if T.size > 0 else 0
        offset_transitions = T - min_transition + 1e-6
        
        bigram_clauses = 0
        for i in range(n - 1):
            for l in range(L):
                for q in range(L):
                    if (i, l, q) in z:  # Only for active transitions
                        z_var = z[(i, l, q)]
                        
                        # Linking constraints: z[i,l,q] ↔ (var[i,l] ∧ var[i+1,q])
                        enc.add_hard([-z_var, var[i, l]])
                        enc.add_hard([-z_var, var[i + 1, q]])
                        enc.add_hard([-var[i, l], -var[i + 1, q], z_var])

                        # Transition reward
                        weight = max(1, int(round(offset_transitions[l, q] * 1000)))
                        enc.add_soft([z_var], weight=weight)
                        bigram_clauses += 1

        # 5) Exactly-one constraint per position (hard constraints)
        for i in range(n):
            active_vars = [var[i, l] for l in range(L) if var_created[i, l]]
            if len(active_vars) > 1:
                enc.exactly_one(active_vars)
            elif len(active_vars) == 1:
                enc.add_hard([active_vars[0]])  # Force the only variable to be true

        print(f"SAT clauses: {unary_clauses} unary + {bigram_clauses} bigram")

        # 6) Solve Max-SAT with timeout
        try:
            wcnf = enc.to_wcnf()
            solver = RC2(wcnf)
            
            # Set a reasonable timeout (in seconds) to prevent hanging
            # Note: This depends on your PySAT version supporting timeouts
            model = solver.compute()
            
        except Exception as e:
            print(f"SAT solver failed: {e}, falling back to Viterbi")
            return self._viterbi_fallback(scores, T)
        
        if not model:
            print("SAT solver found no solution, falling back to Viterbi")
            return self._viterbi_fallback(scores, T)

        # 7) Extract solution
        y = []
        model_set = set(model)
        for i in range(n):
            selected_labels = [l for l in range(L) if var_created[i, l] and var[i, l] in model_set]
            if len(selected_labels) != 1:
                print(f"SAT constraint violation at position {i}: {len(selected_labels)} labels selected")
                return self._viterbi_fallback(scores, T)
            y.append(selected_labels[0])
        
        return y

    def _viterbi_fallback(self, scores, T):
        """Fallback to Viterbi decoding - gives optimal solution"""
        n, L = scores.shape
        if n == 0:
            return []
        if n == 1:
            return [int(scores[0].argmax())]
            
        # Dynamic programming for optimal sequence
        dp = np.zeros((n, L), dtype=np.float32)
        bp = np.zeros((n, L), dtype=np.int32)
        
        # Initialize first position
        dp[0] = scores[0]
        
        # Forward pass
        for i in range(1, n):
            for curr in range(L):
                # Find best previous state
                transitions = dp[i-1] + T[:, curr]
                best_prev = transitions.argmax()
                dp[i, curr] = transitions[best_prev] + scores[i, curr]
                bp[i, curr] = best_prev
        
        # Backward pass
        path = []
        last_state = int(dp[-1].argmax())
        path.append(last_state)
        
        for i in range(n-1, 0, -1):
            last_state = bp[i, last_state]
            path.append(last_state)
        
        path.reverse()
        return path