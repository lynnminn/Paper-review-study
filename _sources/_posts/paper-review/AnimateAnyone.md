# Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation
[![arXiv](https://img.shields.io/badge/arXiv-2210.02441-b31b1b.svg)](https://arxiv.org/pdf/2311.17117.pdf)
[![Github](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/HumanAIGC/AnimateAnyone?tab=readme-ov-file)

![result_example](./materials/AnimateAnyone/result.gif)
[출처](https://www.aitimes.com/news/articleView.html?idxno=155678)

### 키워드
- Image to Video, Stable Diffusion
- Consistent generation 
- Temporal Attention, Spatial Attention

### Preview이자 전부
- 이미지 생성 모델인 stable diffusion(SD)를 활용해서 비디오 생성
  - 이미지는 2D, 비디오는 시간 차원이 추가된 3D
- 3 가지 타입의 Attention 사용
  - Spatial-Attention, Cross-Attention, Temporal-Attention 사용
- SD 사용하기 위해 텐서의 (차원을 늘리는) repeat, (차원 축을 바꾸는) transpose, reshape를 활용
  - Spatial attention : 차원 늘리기 for 2D -> 3D 데이터 
    </br> [b, h, w, c] -> [b, __"t"__, h, w, c]
  - Cross attention : 추가된 차원을 batch에 합하는 방식 for 3D -> 2D attention
    </br> [b, t, h, w, c] -> [__"b*t"__, h, w, c]
  - Temporal attention: 차원 축 바꾸기 for 시간 축을 고려한 attention 계산 
    </br> [b, t, h, w, c] -> [b, __"c"__, h, w, __"t"__]
- 기존 방식의 문제점인 레퍼런스 이미지 구도 의존성을 줄이는 새로운 attention 결합 방식 제안 