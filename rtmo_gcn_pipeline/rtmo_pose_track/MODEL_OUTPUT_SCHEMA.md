# 모델 출력 데이터 스키마 (JSON)

이 문서는 ST-GCN++ 기반 폭력 탐지 모델의 분석 결과를 다른 시스템(UI, 분석 엔진, 데이터베이스 등)에 전달하기 위한 표준 데이터 구조(Schema)를 정의합니다. 이 스키마는 JSON 형식을 기반으로 하며, 실시간 이벤트 전송, API 응답, 로그 저장 등 다양한 용도로 활용될 수 있습니다.

## 1. 개요

데이터 구조는 크게 두 부분으로 나뉩니다.

1.  **이벤트 요약 (Event Summary)**: 특정 시점(타임스탬프)에 발생한 이벤트의 핵심 정보입니다. 비디오 클립 또는 실시간 스트림의 특정 윈도우에 대한 모델의 최종 판단 결과를 담습니다.
2.  **관찰된 객체 (Observed Objects)**: 해당 이벤트에서 탐지된 모든 인물(Person) 객체들의 상세 정보 리스트입니다. 각 인물의 고유 ID, 시계열 관절 좌표, 바운딩 박스, 개별 점수 등을 포함합니다.

이 구조는 하나의 "이벤트"를 단위로 전송되며, 실시간 스트리밍 환경에서는 특정 시간 간격(예: 1초)마다 생성될 수 있고, 비디오 파일 분석 시에는 각 슬라이딩 윈도우마다 생성될 수 있습니다.

## 2. 전체 데이터 구조 (JSON Schema)

```json
{
  "eventId": "evt-0a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
  "timestamp": "2025-08-06T14:30:00.123Z",
  "sourceInfo": {
    "type": "video_file",
    "sourceId": "V_143.mp4",
    "streamUrl": null,
    "clipStartFrame": 0,
    "clipEndFrame": 100
  },
  "eventSummary": {
    "isFight": true,
    "confidence": 0.92,
    "label": "Fight",
    "totalPersons": 8,
    "rankedPersons": 4
  },
  "observedObjects": [
    {
      "objectId": "person-track-10",
      "label": "Person",
      "rank": 1,
      "compositeScore": 0.297,
      "boundingBoxes": [
        {"frameIndex": 0, "box2d": [x1, y1, x2, y2]},
        {"frameIndex": 1, "box2d": [x1, y1, x2, y2]},
        ...
      ],
      "keypoints": [
        {
          "frameIndex": 0,
          "pose2d": [[x, y, score], [x, y, score], ...], 
          "avgConfidence": 0.85
        },
        {
          "frameIndex": 1,
          "pose2d": [[x, y, score], [x, y, score], ...],
          "avgConfidence": 0.88
        },
        ...
      ]
    },
    ...
  ]
}
```

--- 

## 3. 필드 상세 설명

### 3.1. 최상위 필드

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | :--: | --- |
| `eventId` | `string` | O | 이벤트의 고유 식별자 (UUID 권장). |
| `timestamp` | `string` | O | 이벤트가 발생한 시각 (ISO 8601 형식). |
| `sourceInfo` | `object` | O | 데이터 소스 정보. (상세 구조 아래 참조) |
| `eventSummary` | `object` | O | 이벤트 요약 정보. (상세 구조 아래 참조) |
| `observedObjects`| `array` | O | 관찰된 객체(인물)들의 리스트. |

### 3.2. `sourceInfo` 객체

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | :--: | --- |
| `type` | `string` | O | 데이터 소스의 종류. `video_file` 또는 `stream`. |
| `sourceId` | `string` | O | 비디오 파일명 또는 스트림 ID. |
| `streamUrl` | `string` | X | 스트림의 경우, 접속 가능한 URL. |
| `clipStartFrame`| `integer`| O | 분석된 비디오 클립(윈도우)의 시작 프레임 인덱스. |
| `clipEndFrame` | `integer`| O | 분석된 비디오 클립(윈도우)의 끝 프레임 인덱스. |

### 3.3. `eventSummary` 객체

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | :--: | --- |
| `isFight` | `boolean` | O | 폭력 이벤트 여부. 모델의 최종 판단 결과. |
| `confidence` | `float` | O | `isFight` 판단에 대한 신뢰도 점수 (0.0 ~ 1.0). |
| `label` | `string` | O | 사람이 읽을 수 있는 레이블 (예: "Fight", "NonFight"). |
| `totalPersons` | `integer` | O | 해당 클립에서 탐지된 총 인원 수. |
| `rankedPersons`| `integer`| O | `observedObjects` 리스트에 포함된, 랭킹이 매겨진 인원 수. |

### 3.4. `observedObjects` 배열 요소 (개별 인물 객체)

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | :--: | --- |
| `objectId` | `string` | O | 객체의 고유 식별자. `label-trackId` 형식 권장 (예: `person-track-10`). |
| `label` | `string` | O | 객체의 클래스 레이블 (예: "Person"). |
| `rank` | `integer` | O | 이벤트 내에서 해당 인물의 중요도 순위 (1부터 시작). |
| `compositeScore` | `float` | O | `rank`를 결정하는 데 사용된 종합 점수. |
| `boundingBoxes`| `array` | O | 프레임별 2D 바운딩 박스 정보 리스트. (상세 구조 아래 참조) |
| `keypoints` | `array` | O | 프레임별 2D 포즈(관절) 정보 리스트. (상세 구조 아래 참조) |

#### 3.4.1. `boundingBoxes` 배열 요소

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | :--: | --- |
| `frameIndex` | `integer` | O | 클립 내의 상대적 프레임 인덱스 (0부터 시작). |
| `box2d` | `array` | O | `[x_min, y_min, x_max, y_max]` 형식의 좌표. |

#### 3.4.2. `keypoints` 배열 요소

| 필드 | 타입 | 필수 | 설명 |
| --- | --- | :--: | --- |
| `frameIndex` | `integer` | O | 클립 내의 상대적 프레임 인덱스. |
| `pose2d` | `array` | O | 17개 관절의 `[x, y, confidence]` 정보 리스트. `[[x0,y0,c0], [x1,y1,c1], ...]` |
| `avgConfidence`| `float` | O | 해당 프레임의 모든 관절 신뢰도 점수 평균. |

## 4. 사용 예시

-   **실시간 UI**: UI는 `isFight`, `confidence`를 사용하여 경고를 표시하고, `observedObjects`의 `boundingBoxes`와 `keypoints`를 받아와 화면에 실시간으로 오버레이할 수 있습니다.
-   **데이터베이스 저장**: 이벤트가 발생할 때마다 이 JSON 객체 전체를 NoSQL 데이터베이스(예: MongoDB)에 저장하여 나중에 상세 분석이나 검색에 활용할 수 있습니다.
-   **분석 엔진**: 분석 엔진은 `eventId`를 기준으로 이벤트를 식별하고, `observedObjects`의 시계열 데이터를 받아 다른 통계 분석이나 패턴 인식 작업을 수행할 수 있습니다.
