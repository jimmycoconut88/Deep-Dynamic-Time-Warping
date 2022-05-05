from .lib import *

@jit(nopython = True)
def compute_warp_feature(scores):
  F = scores.shape[0]
  C = scores.shape[1]
  N = scores.shape[2]
  M = scores.shape[3]
  R = np.zeros((F, C, N + 1, M + 1))

  R[:,:, 0, 0] = 0

  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = R[:,:, i - 1, j - 1]
      r1 = R[:,:, i - 1, j]
      r2 = R[:,:, i, j - 1]
      rmax = np.maximum(np.maximum(r0, r1), r2)

      R[:,:, i, j] = scores[:,:, i - 1, j - 1] + rmax
  return R[:,:, 1:,1:]

@jit(nopython = True)
def compute_warp_feature_path(warped):
  F = warped.shape[0]
  C = warped.shape[1]
  N = warped.shape[2]
  M = warped.shape[3]
  masks = np.zeros(warped.shape)
  for feature in range(F):
    for channel in range(C):
      dy, dx = N-1, M-1
      masks[feature,channel, dy,dx] = 1
      while dx > 0 or dy > 0:

        max = -np.inf
        pos = (dy,dx)
        if dx > 0 and dy > 0:
          max = warped[feature][channel][dy-1][dx-1]
          pos = (dy-1,dx-1)

        if dx > 0 and warped[feature][channel][dy][dx-1] > max:

          max = warped[feature][channel][dy][dx-1]
          pos = (dy, dx-1)

        if dy > 0 and warped[feature][channel][dy-1][dx] > max:

          max = warped[feature][channel][dy-1][dx]
          pos = (dy-1,dx)

        masks[feature,channel,pos[0],pos[1]] = 1
        dy =pos[0]
        dx=pos[1]

  return masks


@jit(nopython = True)
def compute_warp_matrix(scores):
  C = scores.shape[0]
  N = scores.shape[1]
  M = scores.shape[2]
  R = np.zeros((C, N + 1, M + 1))

  R[:, 0, 0] = 0

  for j in range(1, M + 1):
    for i in range(1, N + 1):
      r0 = R[:, i - 1, j - 1]
      r1 = R[:, i - 1, j]
      r2 = R[:, i, j - 1]
      rmax = np.maximum(np.maximum(r0, r1), r2)

      R[:, i, j] = scores[:, i - 1, j - 1] + rmax
  return R[:, 1:,1:]

@jit(nopython = True)
def compute_warp_path(warped):

  C = warped.shape[0]
  N = warped.shape[1]
  M = warped.shape[2]
  masks = np.zeros(warped.shape)
  for channel in range(C):
    dy, dx = N-1, M-1
    masks[channel, dy,dx] = 1
    while dx > 0 or dy > 0:

      max = -np.inf
      pos = (dy,dx)
      if dx > 0 and dy > 0:
        max = warped[channel][dy-1][dx-1]
        pos = (dy-1,dx-1)

      if dx > 0 and warped[channel][dy][dx-1] > max:

        max = warped[channel][dy][dx-1]
        pos = (dy, dx-1)

      if dy > 0 and warped[channel][dy-1][dx] > max:

        max = warped[channel][dy-1][dx]
        pos = (dy-1,dx)

      masks[channel,pos[0],pos[1]] = 1
      dy =pos[0]
      dx=pos[1]

  return masks