# /workspace/mmpose/mmpose/evaluation/metrics/custom_coco_metric.py

import numpy as np
from xtcocotools.cocoeval import COCOeval
from typing import Optional, List

from mmpose.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class CustomCocoMetric(CocoMetric):
    """Custom COCO metric to exclude specific keypoints from evaluation."""

    def __init__(self,
                 excluded_kpt_indices: Optional[List[int]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.excluded_kpt_indices = excluded_kpt_indices
        if self.excluded_kpt_indices:
            print(f"INFO: CustomCocoMetric will exclude keypoints "
                  f"{self.excluded_kpt_indices} from evaluation.")


    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """
        Override the parent class's evaluation function to modify sigmas
        BEFORE creating the COCOeval object.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)

        # ==================== 핵심 수정 로직 시작 ====================
        # 1. 원본 시그마 값을 데이터셋 메타 정보에서 가져온다.
        sigmas = np.array(self.dataset_meta['sigmas'])

        # 2. 만약 'excluded_kpt_indices' 인자가 주어졌다면, 시그마 배열을 직접 수정한다.
        if self.excluded_kpt_indices is not None:
            # 사용자가 지정한 '제외할 인덱스'의 시그마 값을 0으로 설정한다.
            for idx in self.excluded_kpt_indices:
                if 0 <= idx < len(sigmas):
                    sigmas[idx] = 0
            
            print("INFO: Modified sigmas for evaluation:", sigmas)

        # 3. 수정된 시그마 배열을 사용하여 COCOeval 객체를 생성한다.
        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas,
                             self.use_area)
        # ===================== 핵심 수정 로직 끝 =====================
        
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == 'keypoints_crowd':
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)',
                'AP(M)', 'AP(H)'
            ]
        else:
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                'AR .75', 'AR (M)', 'AR (L)'
            ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str