import cv2
import numpy as np
import scipy.spatial.distance as dist
import math
import sklearn.cluster as clust
import pandas as pd
import matplotlib.pyplot as plt

file = "xep228.png"
color_img = cv2.imread(file)
white_board = np.full(color_img.shape, 255, dtype=color_img.dtype)
gray_img = cv2.imread(file, 0)
(threshold, bin_img) = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
bin_img = ~bin_img
# cv2.imshow("original bin", bin_img)
# kernel = np.ones((5, 5), np.uint8)
# dilated = cv2.dilate(bin_img, kernel, iterations=1)
# cv2.imshow("dilated", dilated)
# cv2.imwrite("dilated.jpg", dilated)

def get_vertices(lineset):
	return np.concatenate((lineset[:, 0, 0:2], lineset[:, 0, 2:4]))

def get_angles(lineset):
	vectors = lineset[:, 0, 0:2] - lineset[:, 0, 2:4]
	return np.arctan2(vectors[:, 1], vectors[:, 0])

def get_centers(lineset):
	return (lineset[:, 0, 0:2] + lineset[:, 0, 2:4])//2

def draw_lines(lineset):
	for x1, y1, x2, y2 in lineset[:, 0, :]:
		cv2.line(white_board, (x1, y1), (x2, y2), (0, 255, 0), 2)

def draw_vertices(vertices, radixes, colors):
	for vertex, radius , color in zip(vertices, radixes, colors):
		tup_col = tuple(map(int, color))
		cv2.circle(white_board, tuple(vertex), radius, tup_col, -1)

def distance(a1, a2):
	euklidian = math.sqrt((a1[0]-a2[0])**2 + (a1[1]-a2[1])**2)
	angle = np.abs(a1[2] - a2[2])
	return euklidian*2 + (angle*50)

def get_distmat(lineset):
	centers = get_centers(lineset)
	angles = get_angles(lineset)
	features = np.array([centers[:,0], centers[:,1], angles])
	distmat = dist.pdist(features.transpose(), distance)
	return dist.squareform(distmat)

def recount(angle, centers):
	cos = math.sin(angle)
	sin = math.cos(angle)
	return np.array([centers[:, 0]*sin + centers[:, 1]*cos, centers[:, 0]*cos - centers[:, 1]*sin]).transpose()

lines = cv2.HoughLinesP(bin_img, 1, 1*(np.pi/180), 30, 30, 10)
#pixel accuracy, angle accuracy, min points in line, min line len, max dist between lines
print(f"i see {lines.shape[0]} lines")

angles = get_angles(lines)
angle_distmat = dist.pdist(np.array([angles]).transpose(), lambda x1, x2: np.abs(x1 - x2))
angle_distmat = dist.squareform(angle_distmat)
angle_distmat = angle_distmat*(180/(np.pi*6))
print(f"max dist is {np.max(angle_distmat)}")
angle_based = clust.DBSCAN(metric="precomputed").fit(angle_distmat)

labels = angle_based.labels_
core = angle_based.core_sample_indices_

print(f"i see {len(set(labels))} clusters")
tup_labels = list(labels)
centers = get_centers(lines)
bd = clust.DBSCAN(eps=5, min_samples = 2)
for label in set(labels):
	indices = np.where(labels == label)[0]
	angle = np.mean(angles[indices])
	subset = centers[indices]
	recounted = recount(angle, subset)
	print(recounted.shape)
	out = bd.fit(np.array([recounted[:, 1]]).transpose())
	print(out.labels_.shape)
	for or_ind, sublabel in zip(indices, out.labels_):
		tup_labels[or_ind] = (labels[or_ind], sublabel)

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')

# plt.scatter(centers[indices][:, 0], centers[indices][:, 1], c="blue")
# plt.scatter(recounted[:, 1], recounted[:, 0], c = "red")

# plt.eventplot(recounted[:, 1], orientation='horizontal', colors='b')
# plt.axis("off")

# x = np.linspace(0, 400, 400)
# y = x*math.tan(np.mean(angles[indices])) + 300
# plt.scatter(x, y, c="green")
# plt.show()

new_labels = pd.factorize(tup_labels)[0]

colors = np.random.randint(256, size=(len(set(new_labels)), 3))
vertex_colors = colors[new_labels]

print(f"old {len(set(labels))} new {len(set(new_labels))}")
vertex_radixes = np.full(new_labels.shape, 3)
vertex_radixes[core] = 7


draw_lines(lines)
draw_vertices(get_centers(lines), vertex_radixes, vertex_colors)

# cv2.imshow("color", white_board)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
cv2.imwrite("both.jpg", white_board)

vertex_colors = colors[labels]
draw_vertices(get_centers(lines), vertex_radixes, vertex_colors)
cv2.imwrite("angle.jpg", white_board)
