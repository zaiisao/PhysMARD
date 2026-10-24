"""Structured foot-contact VAE and HumanML3D prediction interface."""

from .inference import load_model, predict_contacts

__all__ = ["load_model", "predict_contacts"]
