import matplotlib.pyplot as plt
from lib import pose_detection 

d_frame = pose_detection.get_frame_dist()

count = 0
total_vid = 0
for k, v in d_frame.items():
    total_vid += v

threshold = 0.95

total_thresh_vid = int(total_vid * threshold) + 1

print(total_thresh_vid)

for k, v in d_frame.items():
    count += v
    print(k, count)
    if count >= total_thresh_vid:
        #print(str(int(threshold * 100)) + "% dari video yang digunakan memiliki jumlah frame kurang dari sama dengan " + str(k) + " frame")
        break

fig, ax = plt.subplots(figsize=(7, 6))
fig.subplots_adjust(bottom=0.2)
ax.bar(d_frame.keys(), d_frame.values())
ax.set_title("Distribusi frame per video")
ax.set_ylabel("Jumlah video")
ax.set_xlabel("Jumlah frame \n\n" + str(int(threshold * 100)) + "% dari video yang digunakan memiliki jumlah frame kurang dari sama dengan " + str(k) + " frame")
fig.savefig("frame_dist.png")
plt.show()