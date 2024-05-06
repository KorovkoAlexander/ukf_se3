import numpy as np

def Skew3(vec3, m = 0):
	return np.array([
		[m,       -vec3[2],   vec3[1]],
		[vec3[2],        m,  -vec3[0]],
		[-vec3[1], vec3[0],        m]])


def Skew6(vec6, m = 0):
	T = np.zeros((4, 4))
	T[:3, :3] = Skew3(vec6[:3], m)
	T[:3, -1] = vec6[3:]
	return T

def inverse_pose(mat4):
	R = mat4[:3, :3]
	t = mat4[:3, -1]

	T = np.eye(4)
	T[:3, :3] = R.transpose()
	T[:3, -1] = - np.matmul(R.transpose(), t)
	return T
    
def Exp6(twist6):
	if np.linalg.norm(twist6) < 1e-2 :
		return np.eye(4) + Skew6(twist6)

	T = np.eye(4)

	w = twist6[:3]
	v = twist6[3:]

	theta = max(np.linalg.norm(w), 1e-6)

	c = np.cos(theta)
	s_theta = np.sin(theta) / theta
	n = w / theta

	nxv = np.cross(n, v)

	n = n[:, np.newaxis]

	T[:3, :3] = Skew3(w * s_theta, c) + (1 - c) * np.matmul(n, n.transpose())
	T[:3, -1] = v + ((1 - c) / theta * nxv + (1 - s_theta) * np.cross(n[:, 0], nxv))
	return T

def Exp3(twist3):
	if np.linalg.norm(twist3) < 1e-3 :
		return np.eye(3) + Skew3(twist3)

	theta = np.linalg.norm(twist3)
	c = np.cos(theta)
	s_theta = np.sin(theta) / theta
	n = twist3 / theta

	n = n[:, np.newaxis]

	return Skew3(twist3 * s_theta, c) + (1 - c) * np.matmul(n, n.transpose())

def Log3(R):
	a = R - np.eye(3)
	u, s, vh = np.linalg.svd(a, full_matrices=True)
	wvec = vh[2, :]

	rvec = np.array([
		a[2, 1] - a[1, 2],
		a[0, 2] - a[2, 0],
		a[1, 0] - a[0, 1]])
	wmag = np.arctan2(np.dot(rvec, wvec), np.trace(a) + 2)
	return wmag * wvec

def convert_to_positive_semi_definite(matrix, thresh=0):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    positive_eigenvalues = np.maximum(eigenvalues, thresh)
    positive_semi_definite_matrix = np.dot(eigenvectors, np.dot(np.diag(positive_eigenvalues), np.linalg.inv(eigenvectors)))
    return positive_semi_definite_matrix

def Log6(mat4):

	a = mat4[:3, :3] - np.eye(3)
	u, s, vh = np.linalg.svd(a, full_matrices=True)
	wvec = vh[2, :]

	rvec = np.array([
		a[2, 1] - a[1, 2],
		a[0, 2] - a[2, 0],
		a[1, 0] - a[0, 1]])
	wmag = np.arctan2(np.dot(rvec, wvec), np.trace(a) + 2)

	wmag_2 = wmag / 2
	t = mat4[:3, -1]
	wxt = np.cross(wvec, t)

	v = t - wmag_2 * wxt
	if np.abs(wmag) > 1e-7:
		v += (1 - wmag_2 / np.tan(wmag_2)) * np.cross(wvec, wxt)

	result = np.zeros(6)
	result[:3] = wmag * wvec
	result[3:] = v
	return result