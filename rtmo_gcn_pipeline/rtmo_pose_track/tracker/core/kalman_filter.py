#!/usr/bin/env python3
"""
Enhanced Kalman Filter for tracking
MMTracking의 KalmanFilter를 참고하여 향상된 버전 구현
"""

import numpy as np
import scipy.linalg


class EnhancedKalmanFilter:
    """
    향상된 칼만 필터 for 2D 바운딩 박스 트래킹
    MMTracking의 구현을 기반으로 개선
    """
    
    def __init__(self, center_only=False, dt=1.0):
        """
        Args:
            center_only: True이면 중심점만 추적, False이면 전체 박스 추적
            dt: 시간 간격 (기본값 1.0)
        """
        self.center_only = center_only
        self.dt = dt
        
        # Chi-square 분포의 95% 신뢰구간 임계값들
        self.chi2inv95 = {
            1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
            6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
        }
        
        # Gating threshold 설정
        if self.center_only:
            self.gating_threshold = self.chi2inv95[2]  # x, y만
        else:
            self.gating_threshold = self.chi2inv95[4]  # x, y, a, h
        
        # 상태 차원 설정 (위치 + 속도)
        self.ndim = 4  # [center_x, center_y, aspect_ratio, height]
        
        # 상태 전이 행렬 F (위치 = 위치 + 속도*dt)
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # 관측 행렬 H (위치만 관측)
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # 노이즈 가중치 설정 (MMTracking 기본값)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement):
        """
        새로운 트랙을 위한 초기 상태 생성
        
        Args:
            measurement: [center_x, center_y, aspect_ratio, height]
            
        Returns:
            (mean, covariance): 초기 평균과 공분산 행렬
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        # 초기 공분산 설정
        std = [
            2 * self._std_weight_position * measurement[3],  # center_x
            2 * self._std_weight_position * measurement[3],  # center_y  
            1e-2,                                           # aspect_ratio
            2 * self._std_weight_position * measurement[3],  # height
            10 * self._std_weight_velocity * measurement[3], # velocity_x
            10 * self._std_weight_velocity * measurement[3], # velocity_y
            1e-5,                                           # velocity_aspect
            10 * self._std_weight_velocity * measurement[3]  # velocity_height
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        예측 단계 - 다음 상태 예측
        
        Args:
            mean: 현재 상태 평균
            covariance: 현재 공분산 행렬
            
        Returns:
            (mean, covariance): 예측된 평균과 공분산
        """
        # 프로세스 노이즈 설정
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
        
        # 예측 수행
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """
        상태를 측정 공간으로 투영
        
        Args:
            mean: 상태 평균
            covariance: 공분산 행렬
            
        Returns:
            (mean, covariance): 투영된 평균과 공분산
        """
        # 측정 노이즈 설정
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        # 투영 수행
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + innovation_cov
    
    def multi_predict(self, mean, covariance):
        """
        여러 단계 미래 예측 (추가 기능)
        
        Args:
            mean: 상태 평균
            covariance: 공분산 행렬
            
        Returns:
            (mean, covariance): 여러 단계 예측 결과
        """
        return self.predict(mean, covariance)
    
    def update(self, mean, covariance, measurement):
        """
        업데이트 단계 - 측정값으로 상태 보정
        
        Args:
            mean: 예측 상태 평균
            covariance: 예측 공분산 행렬
            measurement: 측정값 [center_x, center_y, aspect_ratio, height]
            
        Returns:
            (mean, covariance): 업데이트된 평균과 공분산
        """
        # 측정 공간으로 투영
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Cholesky 분해를 사용한 효율적인 역행렬 계산
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, overwrite_a=False, check_finite=False)
        
        # 칼만 게인 계산
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), 
            np.dot(covariance, self._update_mat.T).T,
            overwrite_b=False, check_finite=False).T
        
        # 혁신(innovation) 계산
        innovation = measurement - projected_mean
        
        # 상태 업데이트
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements,
                       only_position=False):
        """
        Gating을 위한 마할라노비스 거리 계산
        
        Args:
            mean: 상태 평균
            covariance: 공분산 행렬
            measurements: 측정값들 shape (N, 4)
            only_position: True이면 위치만 고려
            
        Returns:
            거리 배열 shape (N,)
        """
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        # Cholesky 분해
        chol_factor, lower = scipy.linalg.cho_factor(
            covariance, lower=True, overwrite_a=False, check_finite=False)
        
        # 마할라노비스 거리 계산
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            chol_factor, d.T, lower=lower, overwrite_b=False, check_finite=False)
        
        return np.sum(z * z, axis=0)