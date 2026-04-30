"""Tests for src.utils.views."""

import numpy as np
import pytest

import src.utils.views as views


def test_show_frames():
    video = (np.random.rand(512, 256, 256) * 255).astype(np.uint8)
    views.show_frames(video)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])