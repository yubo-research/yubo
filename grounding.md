# Grounding / Hard constraints

## UHD perturbators: avoid O(D) allocations

UHD perturbators **must not allocate O(D) memory** (no full-size dense noise tensors, and no full-size masks) during `perturb()` / `unperturb()`.

Instead, perturbations should be applied **in place** using a streaming/chunked approach (or sparse-index updates) so memory overhead is \(O(1)\) or proportional to the number of perturbed coordinates (nnz), not the full parameter dimension \(D\).
