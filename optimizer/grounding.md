# UHD

Perturbator **memory and allocation** rules (no \(O(D)\) dense noise or full masks in `perturb()` / `unperturb()`, streaming or sparse updates instead) are specified in the repository root [`grounding.md`](../grounding.md). Read that file before changing perturbator implementations.

UHD optimizers should
 - optimize a `TorchPolicy`, `MLPPolicy`, `LinearPolicy`, or any other policy in our codebase
 - when the underlying policy is an nn.Module, perturb the parameters in-place (to save on copies/memory)
 - optionally support BehavioralEmbedding w/an ENN surrogate
 - make use of SigmaAdapter
