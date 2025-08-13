"""
칼만 필터 구현 - ByteTracker용

mmtracking의 칼만 필터를 참고하여 바운딩 박스 트래킹에 최적화했습니다.
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    바운딩 박스 추적을 위한 간단한 칼만 필터

    8차원 상태 공간 (x, y, a, h, vx, vy, va, vh)에서 바운딩 박스를 추적합니다.
    여기서 (x, y)는 중심 좌표, a는 종횡비, h는 높이이며,
    v는 해당 변수들의 속도입니다.
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # 상태 전이 행렬 생성
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 모션 및 관측 불확실성
        # 큰 불확실성 → 빠른 수렴, 불안정한 추적
        # 작은 불확실성 → 느린 수렴, 안정적인 추적
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        주어진 바운딩 박스 측정값으로 트랙 상태를 생성합니다.

        Args:
            measurement: 바운딩 박스 좌표 (x, y, a, h) - 중심점, 종횡비, 높이

        Returns:
            (mean, covariance): 초기 상태 분포를 반환합니다.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y  
            1e-2,                                            # a
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            1e-5,                                            # va
            10 * self._std_weight_velocity * measurement[3]  # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        다음 시간 단계로 상태 분포를 예측합니다.

        Args:
            mean: 이전 상태의 평균 벡터 (8,)
            covariance: 이전 상태의 공분산 행렬 (8, 8)

        Returns:
            (mean, covariance): 예측된 상태 분포
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        상태 공간을 측정 공간으로 투영합니다.

        Args:
            mean: 상태 평균 벡터 (8,)
            covariance: 상태 공분산 행렬 (8, 8)

        Returns:
            (mean, covariance): 측정 공간에서의 평균과 공분산
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        여러 객체 상태를 한 번에 예측합니다.

        Args:
            mean: 상태 평균 벡터들 (N, 8)
            covariance: 상태 공분산 행렬들 (N, 8, 8)

        Returns:
            (mean, covariance): 예측된 상태 분포들
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        칼만 필터 업데이트 단계를 수행합니다.

        Args:
            mean: 예측된 상태 평균 (8,)
            covariance: 예측된 상태 공분산 (8, 8)
            measurement: 바운딩 박스 측정값 (4,)

        Returns:
            (mean, covariance): 업데이트된 상태 분포
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor = scipy.linalg.cholesky(projected_cov, lower=True)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, True), np.dot(covariance, self._update_mat.T).T).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, 
                       measurements: np.ndarray, only_position: bool = False) -> np.ndarray:
        """
        상태 분포와 측정값들 사이의 게이팅 거리를 계산합니다.

        Args:
            mean: 상태 평균 벡터 (8,)
            covariance: 상태 공분산 행렬 (8, 8)
            measurements: 측정값들 (N, 4)
            only_position: True면 위치만 고려, False면 모든 좌표 고려

        Returns:
            각 측정값에 대한 제곱 마할라노비스 거리 (N,)
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        return np.sum(z * z, axis=0)  # 제곱 마할라노비스 거리