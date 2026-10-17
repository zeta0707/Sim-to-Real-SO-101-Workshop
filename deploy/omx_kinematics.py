# SPDX-License-Identifier: Apache-2.0
#
# OMX-F 순기구학 (numpy). 값은 assets/usd/omx_f/omx_f.usd 의 관절 정의에서 읽었다.
#   관절 i 의 부모 프레임 기준 위치(localPos0) / 회전(localRot0) / 축, 그리고 ee 오프셋(omx_env_cfg.py ee_frame).
# 검증: 시뮬레이션 기준값(ref_r10.npz)의 ee_pos 와 99% 샘플에서 5e-7 m 이내로 일치한다.
import numpy as np


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _T(R=None, p=(0.0, 0.0, 0.0)):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = p
    return M


EE_OFFSET = (0.09193, -0.0016, 0.0)  # link5 기준


def fk(q):
    """q: 관절각 rad (최소 5개: joint1..joint5). 반환: 4x4 (로봇 베이스 기준 ee 프레임)."""
    M = _T(None, (-0.01125, 0.0, 0.034)) @ _T(_rz(q[0]))
    M = M @ _T(None, (0.0, 0.0, 0.0635)) @ _T(_ry(q[1]))
    M = M @ _T(None, (0.0415, 0.0, 0.11315)) @ _T(_ry(q[2]))
    M = M @ _T(None, (0.162, 0.0, 0.0)) @ _T(_ry(q[3]))
    M = M @ _T(_rx(-np.pi / 2), (0.0287, 0.0, 0.0)) @ _T(_rx(q[4]))
    return M @ _T(None, EE_OFFSET)


def rotmat_to_quat_wxyz(R):
    """3x3 회전행렬 -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)
