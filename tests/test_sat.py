import numpy as np
from src.sat_infer import SATInfer
# Test the fix
def test_sat():
    """Test case with detailed debugging"""
    sat = SATInfer()
    
    # Test case 1: The original failing test
    print("=== Test Case 1: Zero unary scores, high transition costs ===")
    L = 3
    scores = np.zeros((2, L))  # All unary scores are 0
    T = np.array([[5, 0, 0],   # High cost for staying in same state  
                  [0, 5, 0],   # Prefer transitions to different states
                  [0, 0, 5]], dtype=np.float32)
    
    print(f"Scores:\n{scores}")
    print(f"Transition matrix T:\n{T}")
    
    # Expected: should prefer [0,1] because T[0,1] = 0 is better than T[0,0] = 5
    # But wait - that's wrong! T[0,0] = 5 means high REWARD for staying in state 0
    # So the optimal should actually be [0,0] or [1,1] or [2,2]
    
    result = sat.argmax(scores, T)
    print(f"SAT result: {result}")
    
    # Verify with manual calculation
    if len(result) == 2:
        total_score = scores[0, result[0]] + scores[1, result[1]] + T[result[0], result[1]]
        print(f"Total score for {result}: {total_score}")
    
    # Test all possible sequences manually
    print("All possible scores:")
    for i in range(L):
        for j in range(L):
            score = scores[0, i] + scores[1, j] + T[i, j]
            print(f"  [{i}, {j}]: {score}")
    
    print()
    
    # Test case 2: Non-zero unary scores
    print("=== Test Case 2: Non-zero unary scores ===")
    scores2 = np.array([[1, 0, 0],    # Prefer label 0 at position 0
                        [0, 1, 0]])   # Prefer label 1 at position 1
    
    print(f"Scores:\n{scores2}")
    print(f"Transition matrix T:\n{T}")
    
    result2 = sat.argmax(scores2, T)
    print(f"SAT result: {result2}")
    
    # Manual verification
    print("All possible scores:")
    for i in range(L):
        for j in range(L):
            score = scores2[0, i] + scores2[1, j] + T[i, j]
            print(f"  [{i}, {j}]: {score}")
    
    # Find optimal manually
    best_score = -float('inf')
    best_seq = None
    for i in range(L):
        for j in range(L):
            score = scores2[0, i] + scores2[1, j] + T[i, j]
            if score > best_score:
                best_score = score
                best_seq = [i, j]
    
    print(f"Manual optimal: {best_seq} with score {best_score}")
    
    if len(result2) == 2:
        sat_score = scores2[0, result2[0]] + scores2[1, result2[1]] + T[result2[0], result2[1]]
        print(f"SAT score: {sat_score}")
        print(f"Optimal: {'✓' if sat_score == best_score else '✗'}")
    
    print()
    
    # Test case 3: Longer sequence
    print("=== Test Case 3: Longer sequence ===")
    scores3 = np.array([[2, 1, 0],    
                        [1, 2, 0],    
                        [0, 1, 2]])
    
    result3 = sat.argmax(scores3, T)
    print(f"Longer sequence result: {result3}")
    
    return result, result2, result3

def compare_with_viterbi():
    """Compare SAT results with Viterbi algorithm"""
    print("=== Comparing SAT with Viterbi ===")
    
    def viterbi(scores, T):
        n, L = scores.shape
        if n == 1:
            return [int(scores[0].argmax())]
            
        dp = np.zeros((n, L))
        bp = np.zeros((n, L), dtype=int)
        
        dp[0] = scores[0]
        
        for i in range(1, n):
            for j in range(L):
                candidates = dp[i-1] + T[:, j]
                bp[i, j] = candidates.argmax()
                dp[i, j] = candidates[bp[i, j]] + scores[i, j]
        
        # Backtrack
        path = [int(dp[-1].argmax())]
        for i in range(n-1, 0, -1):
            path.append(bp[i, path[-1]])
        path.reverse()
        return path
    
    sat = SATInfer()
    
    # Test several cases
    test_cases = [
        (np.zeros((2, 3)), np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float32)),
        (np.array([[1, 0, 0], [0, 1, 0]]), np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float32)),
        (np.array([[2, 1, 0], [1, 2, 0], [0, 1, 2]]), np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float32)),
    ]
    
    for i, (scores, T) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        sat_result = sat.argmax(scores, T)
        viterbi_result = viterbi(scores, T)
        
        print(f"  SAT:     {sat_result}")
        print(f"  Viterbi: {viterbi_result}")
        print(f"  Match:   {'✓' if sat_result == viterbi_result else '✗'}")
        
        # Calculate scores
        if len(sat_result) > 0:
            sat_score = sum(scores[t, sat_result[t]] for t in range(len(sat_result)))
            if len(sat_result) > 1:
                sat_score += sum(T[sat_result[t], sat_result[t+1]] for t in range(len(sat_result)-1))
            
            vit_score = sum(scores[t, viterbi_result[t]] for t in range(len(viterbi_result)))
            if len(viterbi_result) > 1:
                vit_score += sum(T[viterbi_result[t], viterbi_result[t+1]] for t in range(len(viterbi_result)-1))
            
            print(f"  SAT score:     {sat_score}")
            print(f"  Viterbi score: {vit_score}")


if __name__ == "__main__":
    test_sat()
    print("\n" + "="*50)
    compare_with_viterbi()