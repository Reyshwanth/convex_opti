# Rey Ellipsoid Method - Time Complexity Analysis

## Overall Complexity

$$O(n^2 \cdot k + (n^2 \cdot k)^3)$$

Where:
- $n$ = number of beacons (20 in the default configuration)
- $k$ = samples per intersection circle (8 in the default configuration)

## Breakdown by Step

| Step | Operation | Complexity |
|------|-----------|------------|
| 1 | **Pairwise intersection check** | $O(n^2)$ — checking all $\binom{n}{2}$ beacon pairs |
| 2 | **Circle sampling** | $O(n^2 \cdot k)$ — up to $n^2$ intersections, $k$ points each |
| 3 | **Weight computation** | $O(n^2 \cdot k)$ — one weight per sampled point |
| 4 | **Weighted MVEE (SDP)** | $O(m^3)$ where $m = n^2 \cdot k$ sampled points |

## The Bottleneck: SDP Solver

The **Semidefinite Programming** solver dominates the runtime. For the default setup:
- $m = 171 \times 8 = 1368$ points
- SDP with 1368 constraints in 3D takes ~11 seconds

The SDP complexity is roughly $O(m^3)$ for interior-point methods, though modern solvers like SCS use first-order methods with better practical scaling (~$O(m^2)$ per iteration).

## Practical Scaling

| Beacons ($n$) | Max Points ($m$) | Approx Runtime |
|---------------|------------------|----------------|
| 20 | ~1,500 | ~10s |
| 50 | ~10,000 | ~minutes |
| 100 | ~40,000 | ~hours |

## Optimization Strategies

To scale better, consider:

1. **Reduce $k$** — fewer samples per intersection circle
2. **Subsample intersections** — only use highest-weight pairs
3. **Use iterative/approximate MVEE algorithms** — e.g., Khachiyan's algorithm with early stopping
4. **Parallel sampling** — intersection checks and circle sampling are embarrassingly parallel
