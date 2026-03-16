# -*- coding: utf-8 -*-
"""Kraken utils for OCR/HTR engine"""

import streamlit as st

from kraken.lib import (vgsl,
                        models)
from kraken import (blla,
                    rpred)
from PIL import Image

@st.cache_data(show_spinner=False)
def load_model_seg(model_path: str) -> vgsl.TorchVGSLModel:
    """Load a segmentation model"""
    return vgsl.TorchVGSLModel.load_model(model_path)

@st.cache_data(show_spinner=False)
def load_model_rec(model_path: str):
    """Load a recognition model"""
    return models.load_any(model_path)


def segment_image(image: Image, model_seg: vgsl.TorchVGSLModel):
    """Segment an image"""
    return blla.segment(image, model=model_seg)


def recognize_text(model, image: Image, baseline_seg):
    """Recognize text in an image"""
    return rpred.rpred(network=model, im=image, bounds=baseline_seg, pad=16, bidi_reordering=True)
