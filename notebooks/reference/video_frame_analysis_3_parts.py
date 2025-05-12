# %%
import numpy as np
from video_import import *
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.filters.thresholding import threshold_li, threshold_otsu
from skimage.filters import unsharp_mask, rank
from skimage.segmentation import mark_boundaries, chan_vese, slic, watershed
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.graph import rag_mean_color, cut_normalized, merge_hierarchical
from skimage.exposure import adjust_gamma, adjust_log, equalize_hist, equalize_adapthist
import cv2 
import seaborn as sns
import pandas as pd
import math
import matplotlib.gridspec as gridspec

# %%
data = import_avi("/Users/juanliyau/Downloads/video_1.avi")
coordinates_file = pd.ExcelFile("xy coordinates for knee imaging 0913.xlsx")

# %%
sheet_name = "8.29 3rd"
coordinates = pd.read_excel(coordinates_file, sheet_name)
coordinates = coordinates[['Frame Number','Points','X','Y']]
frames_to_analyze = np.unique(coordinates["Frame Number"])
frames_to_analyze = frames_to_analyze[~np.isnan(frames_to_analyze)]
frames_to_analyze = frames_to_analyze.astype(int)
coordinates["Frame Number"] = np.repeat(frames_to_analyze,4)

# %%
def newLine(p1, p2):
    m = (p1[1]-p2[1])/(p1[0]-p2[0])
    b = p1[1] - m*p1[0]
    return m,b

fig, ax = plt.subplots(1,3,figsize=(20,10))
fig.patch.set_facecolor('white')

left_part_means = []
middle_part_means = []
right_part_means = []

left_part_sum = []
middle_part_sum = []
right_part_sum = []
#other_part_sum = []

left_part_mins = []
middle_part_mins = []
right_part_mins= []

left_part_max = []
middle_part_max = []
right_part_max= []

left_part_stdev = []
middle_part_stdev = []
right_part_stdev= []

left_part_count = []
middle_part_count = []
right_part_count= []

left_part_q1 = []
middle_part_q1 = []
right_part_q1= []

left_part_median = []
middle_part_median = []
right_part_median= []

left_part_q3 = []
middle_part_q3 = []
right_part_q3= []

# %%
for frame_preview in frames_to_analyze:
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[0].set_facecolor('white')
    ax[1].set_facecolor('white')
    ax[2].set_facecolor('white')
    frame_preview = frame_preview - 1
    print(frame_preview)
    frame = data[:,:,:,frame_preview]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sigma = 3
    frame_smooth = ndi.gaussian_filter(frame, sigma) # >>> Used Gaussian filter with sigma 3 and window size (25,25)

    contrasted_image_log = adjust_log(frame_smooth, 1) # >>> Major difference: he log-normalizes the intensity

    threshold_value_otsu = threshold_otsu(contrasted_image_log)
    binary_image_otsu = contrasted_image_log > threshold_value_otsu
    binary_image_otsu = binary_image_otsu.astype(int)

    points = coordinates[coordinates["Frame Number"] == frame_preview+1][['Points',"X","Y"]]

    point1 = (round(points[points["Points"]==1]["X"].iloc[0]), round(points[points["Points"]==1]["Y"].iloc[0]))
    point2 = (round(points[points["Points"]==2]["X"].iloc[0]), round(points[points["Points"]==2]["Y"].iloc[0]))
    point3 = (round(points[points["Points"]==3]["X"].iloc[0]), round(points[points["Points"]==3]["Y"].iloc[0]))
    point4 = (round(points[points["Points"]==4]["X"].iloc[0]), round(points[points["Points"]==4]["Y"].iloc[0]))
    line1 = [[point1[0], point2[0]], [point1[1], point2[1]]]
    line2 = [[point3[0], point4[0]], [point3[1], point4[1]]]
    mask = np.zeros((binary_image_otsu.shape[0],binary_image_otsu.shape[1]))

    m1,b1 = newLine(point1,point2)
    m2,b2 = newLine(point3,point4)

    critical_x1_points = [max(round(-b1/m1), 0), min(round((binary_image_otsu.shape[0]-b1)/m1), binary_image_otsu.shape[1])]
    critical_x2_points = [max(round(-b2/m2), 0), min(round((binary_image_otsu.shape[0]-b2)/m2), binary_image_otsu.shape[1])]

    critical_x1_min = min(critical_x1_points)
    critical_x1_max = max(critical_x1_points)
    critical_x2_min = min(critical_x2_points)
    critical_x2_max = max(critical_x2_points)

    if critical_x1_min > 0:
        critical_x1_min = 0
    if critical_x1_max >= binary_image_otsu.shape[1]:
        critical_x1_max = binary_image_otsu.shape[1]-1
    if critical_x2_min < 0:
        critical_x2_min = 0
    if critical_x2_max <= binary_image_otsu.shape[1]:
        critical_x2_max = binary_image_otsu.shape[1]-1
    if m1>=0:
        for x in range(critical_x1_min, critical_x1_max+1):
            y = m1*x+b1
            y = round(y)
            if y>=0:
                mask[y:,x] = 2
            else:
                mask[:,x] = 2
    else:
        for x in range(critical_x1_min, critical_x1_max+1):
            y = m1*x+b1
            y = round(y)
            mask[:y,x] = 2


    if m2<=0:
        for x in range(critical_x2_min, critical_x2_max+1):
            y = m2*x+b2
            y = round(y)
            if y>=0:
                mask[y:,x] = 3
            else:
                mask[:,x] = 3
    else:
        for x in range(critical_x2_min, critical_x2_max+1):
            y = m2*x+b2
            y = round(y)
            if y>=0:
                mask[:y,x] = 3


    binary_image_otsu[(binary_image_otsu==1) & (mask==2)] = 2
    binary_image_otsu[(binary_image_otsu==1) & (mask==3)] = 3

    left_labels_x_indices = np.unique(np.where(binary_image_otsu == 2)[1])
    if len(np.where(np.append(left_labels_x_indices[1:],left_labels_x_indices[-1]+1) - left_labels_x_indices !=1)[0])>0:
        starting_index_to_separate = np.where(np.append(left_labels_x_indices[1:],left_labels_x_indices[-1]+1) - left_labels_x_indices !=1)[0][0] + 1
        indices_that_are_not_left = left_labels_x_indices[starting_index_to_separate:]
        for i in range(indices_that_are_not_left[0], indices_that_are_not_left[-1]+1):
            for j in range(binary_image_otsu.shape[0]):
                if binary_image_otsu[j,i] == 2:
                    binary_image_otsu[j,i] = 1
    '''
    m = mark_boundaries(frame, binary_image_otsu)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    show_frame = np.copy(frame)
    show_frame[np.all(m==[1,1,0], axis=2)] = [255,255,0]

    # Make boudnaries bigger
    show_frame_scaled = np.copy(show_frame)
    scaling = 3
    boundaries = np.where(np.all(show_frame_scaled==[255,255,0], axis=2))
    for i in range(len(boundaries[0])):
        x = boundaries[0][i]
        y = boundaries[1][i]

        for j in range(1,scaling+1):
            # top
            show_frame_scaled[x,y+1*j,:] = [255,255,0]
            # bottom
            show_frame_scaled[x,y-1*j,:] = [255,255,0]
            # right
            show_frame_scaled[x+1*j,y,:] = [255,255,0]
            # left
            show_frame_scaled[x-1*j,y,:] = [255,255,0]
            # top_right
            show_frame_scaled[x+1*j,y+1*j,:] = [255,255,0]
            # top_left
            show_frame_scaled[x-1*j,y+1*j,:] = [255,255,0]
            # bottom_right
            show_frame_scaled[x+1*j,y-1*j,:] = [255,255,0]
            # bottom_left
            show_frame_scaled[x-1*j,y-1*j,:] = [255,255,0]

    ax[0].imshow(frame)
    ax[0].set_title(f"Original Frame #{frame_preview+1}")
    ax[1].imshow(binary_image_otsu, interpolation="nearest")
    ax[1].set_title("4 Points")
    ax[1].scatter(point1[0], point1[1], label="1")
    ax[1].scatter(point2[0], point2[1], label="2")
    ax[1].scatter(point3[0], point3[1], label="3")
    ax[1].scatter(point4[0], point4[1], label="4")
    ax[1].plot(line1[0], line1[1])
    ax[1].plot(line2[0], line2[1])
    ax[2].imshow(show_frame_scaled, cmap="gray")
    ax[2].set_title("3 Parts")


    fig.savefig(f"./frame_plots_3/{sheet_name}/{frame_preview+1}.png", transparent=False, bbox_inches="tight")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    '''
    left_part_means.append(np.mean(frame[binary_image_otsu == 2]))
    middle_part_means.append(np.mean(frame[binary_image_otsu == 1]))
    right_part_means.append(np.mean(frame[binary_image_otsu == 3]))

    left_part_sum.append(np.sum(frame[binary_image_otsu == 2]))
    middle_part_sum.append(np.sum(frame[binary_image_otsu == 1]))
    right_part_sum.append(np.sum(frame[binary_image_otsu == 3]))
    #other_part_sum.append(np.sum(frame[binary_image_otsu == 0]))

    left_part_mins.append(np.min(frame[binary_image_otsu == 2]))
    middle_part_mins.append(np.min(frame[binary_image_otsu == 1]))
    right_part_mins.append(np.min(frame[binary_image_otsu == 3]))

    left_part_max.append(np.max(frame[binary_image_otsu == 2]))
    middle_part_max.append(np.max(frame[binary_image_otsu == 1]))
    right_part_max.append(np.max(frame[binary_image_otsu == 3]))

    left_part_stdev.append(np.std(frame[binary_image_otsu == 2]))
    middle_part_stdev.append(np.std(frame[binary_image_otsu == 1]))
    right_part_stdev.append(np.std(frame[binary_image_otsu == 3]))

    left_part_count.append(len(frame[binary_image_otsu == 2]))
    middle_part_count.append(len(frame[binary_image_otsu == 1]))
    right_part_count.append(len(frame[binary_image_otsu == 3]))

    left_part_q1.append(np.quantile(frame[binary_image_otsu == 2], q=0.25))
    middle_part_q1.append(np.quantile(frame[binary_image_otsu == 1], q=0.25))
    right_part_q1.append(np.quantile(frame[binary_image_otsu == 3], q=0.25))

    left_part_median.append(np.quantile(frame[binary_image_otsu == 2], q=0.5))
    middle_part_median.append(np.quantile(frame[binary_image_otsu == 1], q=0.5))
    right_part_median.append(np.quantile(frame[binary_image_otsu == 3], q=0.5))

    left_part_q3.append(np.quantile(frame[binary_image_otsu == 2], q=0.75))
    middle_part_q3.append(np.quantile(frame[binary_image_otsu == 1], q=0.75))
    right_part_q3.append(np.quantile(frame[binary_image_otsu == 3], q=0.75))

#%%
d = [i for i in range(len(frames_to_analyze)) if frames_to_analyze[i]==630]
# %%
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor('white')
#xaxis = [i for i in range(len(frames_to_analyze))]
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

ax0 = plt.subplot(gs[0, :])
ax0.plot(frames_to_analyze, left_part_sum, label="left", color = "g")
ax0.plot(frames_to_analyze, middle_part_sum, label="middle", color = "b")
ax0.plot(frames_to_analyze, right_part_sum, label="right", color="y")
ax0.axvline(x=117, color='k', linestyle='--')
ax0.set_title("Sum of Pixel Intensities for all 3 Parts")
ax0.set_xlabel("Frame Index")
ax0.set_ylabel("Sum of Pixel Intensities")
#ax0.set_xticks(xaxis,labels = frames_to_analyze)
ax0.legend()

ax1 = plt.subplot(gs[1, 0])
ax1.plot(frames_to_analyze, left_part_sum, label="left", color = "g")
ax1.set_title("Left Part")

ax2 = plt.subplot(gs[1, 1])
ax2.plot(frames_to_analyze, middle_part_sum, label="middle", color = "b")
ax2.set_title("Middle Part")

ax3 = plt.subplot(gs[1, 2])
ax3.plot(frames_to_analyze, right_part_sum, label="right", color = "y")
ax3.set_title("Right Part")

plt.tight_layout()
plt.show()

fig.savefig(f"./frame_plots_3/{sheet_name}/sum_3_plots.png", transparent=False, bbox_inches="tight")

# %%
fig = plt.figure(figsize=(20,5))
s = []
for i in range(len(frames_to_analyze)):
    s.append(left_part_sum[i]+middle_part_sum[i]+right_part_sum[i]+other_part_sum[i])

plt.plot([i for i in range(len(frames_to_analyze))], s, c="k")
plt.axvline(x=42, color='k', linestyle='--')
plt.title("Total Intensity per Frame")
plt.xlabel("Frame")
plt.ylabel("Intensity")
plt.show()
fig.savefig(f"./frame_plots_3/{sheet_name}/tot_intensity.png", transparent=False, bbox_inches="tight")
# %%
columns = pd.MultiIndex.from_product([['Pixel Intensity Sum', 'Average Pixel Intensity','Pixel Intensity Standard Deviation', "Min Pixel Intensity", "First Quartile Pixel Intensity", "Median Pixel Intensity", "Third Quartile Pixel Intensity", "Max Pixel Intensity"], ['Left Part', 'Middle Part', 'Righ Part']])
rows = pd.MultiIndex.from_product([["Frame"], frames_to_analyze])
stats = np.column_stack((left_part_sum, middle_part_sum, right_part_sum, left_part_means, middle_part_means, right_part_means, left_part_stdev, middle_part_stdev, right_part_stdev, left_part_mins, middle_part_mins, right_part_mins, left_part_q1, middle_part_q1, right_part_q1, left_part_median, middle_part_median, right_part_median, left_part_q3, middle_part_q3, right_part_q3, left_part_max, middle_part_max, right_part_max))
df = pd.DataFrame(stats, columns=columns, index = rows)
print(df)
# %%
with pd.ExcelWriter('./region_data2.xlsx', mode="a") as writer:
    df.to_excel(writer, sheet_name=sheet_name)
    # df2.to_excel(writer, sheet_name='Sheet2', index=False)  # Example for another DataFrame
# %%
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor('white')
#xaxis = [i for i in range(len(frames_to_analyze))]
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

ax0 = plt.subplot(gs[0, :])
ax0.plot(frames_to_analyze, left_part_means, label="left", color = "g")
ax0.plot(frames_to_analyze, middle_part_means, label="middle", color = "b")
ax0.plot(frames_to_analyze, right_part_means, label="right", color="y")
ax0.axvline(x=629, color='k', linestyle='--')
ax0.set_title("Mean of Pixel Intensities for all 3 Parts")
ax0.set_xlabel("Frame Index")
ax0.set_ylabel("Sum of Pixel Intensities")
#ax0.set_xticks(xaxis,labels = frames_to_analyze)
ax0.legend()

ax1 = plt.subplot(gs[1, 0])
ax1.plot(frames_to_analyze, left_part_means, label="left", color = "g")
ax1.set_title("Left Part")

ax2 = plt.subplot(gs[1, 1])
ax2.plot(frames_to_analyze, middle_part_means, label="middle", color = "b")
ax2.set_title("Middle Part")

ax3 = plt.subplot(gs[1, 2])
ax3.plot(frames_to_analyze, right_part_means, label="right", color = "y")
ax3.set_title("Right Part")

plt.tight_layout()
plt.show()

fig.savefig(f"./frame_plots_3/{sheet_name}/mean_3_plots.png", transparent=False, bbox_inches="tight")
# %%
