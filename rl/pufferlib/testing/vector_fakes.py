"""Backward-compatible re-export (prefer ``testing_support.vector_fakes``)."""

from testing_support.vector_fakes import FakePufferVecEnv, FakePufferVecEnvContinuous


__all__ = ["FakePufferVecEnv", "FakePufferVecEnvContinuous"]
