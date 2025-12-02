from utils import views, io

#1358
"1360-1384	1385-1406	1407-1433	1434-1454	1461-1483	1484-1508	1509-1540	1541-1559	1618-1648	1649-1669	1672-1696	1697-1720"

#1342
"62-81	82-100	102-119	123-151	152-171	178-199"

# video = io.load_video("../data/processed/1358_aging_radial_video_N64.npy"); masks = io.load_masks("../data/processed/1358_aging_radial_masks_N64.npy")
# video = io.load_video("../data/processed/1342_aging_radial_video_N64.npy"); masks = io.load_masks("../data/processed/1342_aging_radial_masks_N64.npy")


# v1 = views.draw_mask_boundaries(video, masks)
# views.show_frames(v1)

# views.draw_mask_boundary(video, masks, True)

# video = io.load_mp4(fr"1358N64.mp4")
video = io.load_mp4(fr"1342N64.mp4") 
"Does the normalized heatmap always reflect maximum intensity around segment 57 because it's the largest one?"
"Should consult the unnormalized heatmap to check..."
views.show_frames(video)