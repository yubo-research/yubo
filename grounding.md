# Grounding / Hard constraints

## UHD perturbators: avoid O(D) allocations

UHD perturbators **must not allocate O(D) memory** (no full-size dense noise tensors, and no full-size masks) during `perturb()` / `unperturb()`.

Instead, perturbations should be applied **in place** using a streaming/chunked approach (or sparse-index updates) so memory overhead is \(O(1)\) or proportional to the number of perturbed coordinates (nnz), not the full parameter dimension \(D\).


- Registries should defer complex object creation until an item is selected for use.
- Registries should keep item names in a single place so that humans can read and understand the code more easily.
