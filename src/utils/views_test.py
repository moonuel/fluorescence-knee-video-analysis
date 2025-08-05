import utils.views as views
import numpy as np

def test_show_frames():
    video = (np.random.rand(512,256,256)*255).astype(np.uint8)
    views.show_frames(video)

if __name__ == "__main__":
    test_show_frames()