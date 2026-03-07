
# UHD
UHD optimizers should
 - optimize a `TorchPolicy`, `MLPPolicy`, `LinearPolicy`, or any other policy in our codebase
 - when the underlying policy is an nn.Module, perturb the parameters in-place (to save on copies/memory)
 - optionally support BehavioralEmbedding w/an ENN surrogate
 - make use of SigmaAdapter
