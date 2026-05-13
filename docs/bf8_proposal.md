# BF8 Prototype Proposal

This repo can treat `bf8` as an experimental 8-bit storage format with BF16-like
range goals and lower precision than BF16.

## Goal

The design goal is to keep the dynamic range story closer to BF16 than FP8,
while still reducing storage / bandwidth below BF16.

This is a prototype format, not a native framework dtype.

## First-pass bit layout

- 1 sign bit
- 4 exponent bits
- 3 mantissa bits

That gives 8 total bits.

## Intended behavior

- Store values in BF8.
- Dequantize to BF16, FP16, or FP32 for compute.
- Keep accumulation in a higher precision than BF8.
- Use explicit scaling / conversion rules rather than pretending the runtime
  natively supports BF8.

## Suggested numeric policy

- Round to nearest, ties to even if practical.
- Saturate on overflow.
- Flush underflowed subnormals to zero if that is simpler for the prototype.

## What to test

1. Encode/decode roundtrip error.
2. Saturation and underflow rate on real tensors.
3. Loss / score stability against BF16 and FP8 on a small proxy model.
4. Storage and bandwidth savings.
5. Compatibility with UHD-style subspace perturbation or codec logic.

## Implementation shape

The first implementation should probably be:

- a codec layer
- a small set of conversion helpers
- an experiment flag or config option

It should not require adding a brand-new native dtype to the framework on day
one.
