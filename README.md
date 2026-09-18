# Minimal-Channel EEG-Based Deep Learning Classification for Vascular Mild Cognitive Impairment Screening
### 혈관성 경도인지장애 선별을 위한 최소 채널 뇌파 기반 딥러닝 분류 연구

**강태우¹, 김동빈¹, 박도근¹˒², 손종희³, 원동옥¹˒⁴˒⁵†**

¹ 인공지능융합학부, 한림대학교
² 뇌혈관질환선도연구센터, 한림대학교
³ 신경과, 한림대학교 춘천성심병원
⁴ 의과대학, 한림대학교
⁵ 인구 및 정량보건과학과, 매사추세츠 의과대학

**Taewoo Kang¹, Dongbin Kim¹, Dogeun Park¹˒², Jong-Hee Sohn³, Dong-Ok Won¹˒⁴˒⁵†**

¹ Dept. of Artificial Intelligence Convergence, Hallym University
² Cerebrovascular Disease Research Center, Hallym University
³ Dept. of Neurology, Hallym University Chuncheon Sacred Heart Hospital
⁴ College of Medicine, Hallym University
⁵ Dept. of Population and Quantitative Health Sciences, University of Massachusetts

† E-mail: dongok.won@hallym.ac.kr

한국BCI학회 추계학술대회 (2026.09.30–10.01, 성균관대학교 자연과학캠퍼스 대강당) 제출 초록 (최종본)

---

## Abstract

혈관성 경도인지장애(Vascular Mild Cognitive Impairment, VMCI)는 뇌혈관질환과 관련된 인지기능 저하로,
안정 상태 뇌파(Electroencephalography, EEG)에서 서파화와 alpha 대역 변화가 보고되어 왔다. EEG는
비침습적으로 뇌의 기능적 변화를 측정할 수 있어 인지저하 선별을 위한 유용한 도구로 주목받고 있다. 연구
환경에서 주로 활용되는 다채널 EEG는 높은 공간 해상도를 통해 정밀한 뇌 신호 분석이 가능하지만, 다수의
전극 부착과 검사 준비에 상당한 시간과 전문 인력이 요구되어 1차 의료기관 및 지역사회 임상 선별 환경
적용에는 제약이 따른다. 따라서, 검사의 실용성과 현장 적용성을 함께 고려할 때, 최소 채널 EEG 기반
VMCI 분류 시스템 개발은 임상적으로 중요한 의의를 가진다. 이에 본 연구는 딥러닝 및 정량 뇌파(quantitative
EEG, qEEG) 기반 로지스틱 회귀 모델을 활용하여 최소 채널 EEG 기반 VMCI-정상대조군(Healthy Controls,
HC) 분류 가능성을 평가한다.

데이터셋은 CAUEEG[1]를 활용하였으며, 총 131명(VMCI 61명, HC 70명)으로 구성하여 이진 분류 과제를
진행하였다. 원본 데이터는 19채널 안정 상태 EEG를 활용하여, 0.5–40 Hz 필터링, 독립성분분석 기반 잡음
제거, 불량 채널 탐지·보간 및 평균 재참조를 거쳤으며, 이후 채널별 z-score 정규화 후 4초 길이·2초 간격으로
윈도우를 분할하였다. 분류 정확도 평가를 위해 EEGNet[2]과 5개 주파수 대역의 상대 파워를 이용한 qEEG
기반 로지스틱 회귀 모델이 활용되었으며, 피험자 단위 5-fold 교차검증을 활용하였다.

VMCI–HC 분류과제에서 19채널을 활용한 EEGNet[2]과 qEEG 기반 로지스틱 회귀 모델의 AUC는 각각
0.746±0.090과 0.689±0.098를 보고하였으며, 후두부 2채널(O1, O2)을 활용한 EEGNet과 qEEG 기반
로지스틱 회귀 모델의 AUC는 0.738±0.051과 0.656±0.067를 보고하였다.

본 연구는 VMCI–HC 분류과제에서 최소 채널 EEG 기반 분류 가능성을 보여준다. 특히 연구에서 활용된
다양한 분류 모델들(EEGNet, qEEG 기반 로지스틱 회귀)은 후두부 2채널만으로도 19채널에 준하는 분류
성능을 보고함으로써, 최소 채널의 강건성을 시사한다. 향후, 추가적인 독립 코호트에서 최소 채널 측정
성능 평가가 필요하지만, 본 연구는 소수 채널을 활용한 임상 선별 보조 도구 개발 가능성을 가속화한다.

<sub>(아래는 비공식 영문 요약이며, 실제 제출된 국문 초록을 대체하지 않습니다 / unofficial English summary,
not a substitute for the Korean abstract actually submitted.)</sub>

> Vascular Mild Cognitive Impairment (VMCI) is a cognitive decline associated with cerebrovascular
> disease, with slowing and alpha-band changes reported in resting-state EEG. While multi-channel
> EEG offers high spatial resolution, its lengthy setup and need for trained personnel limit its use
> in primary care and community screening settings. This study evaluates minimal-channel EEG-based
> VMCI–HC classification using EEGNet and a qEEG-based logistic regression model on 131 subjects
> (61 VMCI, 70 HC) from the CAUEEG dataset. Using all 19 channels, EEGNet and qEEG achieved AUCs of
> 0.746±0.090 and 0.689±0.098; using only the two occipital channels (O1, O2), AUCs of 0.738±0.051
> and 0.656±0.067 were achieved — showing that a minimal 2-channel montage can reach performance
> comparable to the full 19-channel set.

## Dataset

- **소스**: CAUEEG 기반 파생 데이터셋 ([Kim et al., 2023, *NeuroImage*](https://doi.org/10.1016/j.neuroimage.2023.120054))
- **표본**: 총 131명 (VMCI 61명, HC 70명) — 이진 분류 과제
- **원본 신호**: 19채널 안정 상태(resting-state) EEG

## Preprocessing

1. 250Hz 리샘플링
2. 채널별 z-score 정규화
3. 4초 윈도우, 2초 stride로 분할

## Models

| 모델 | 입력 | 설명 |
|---|---|---|
| **EEGNet** | raw EEG signal | Lawhern et al. (2018) compact CNN 구조 |
| **qEEG + Logistic Regression** | 5개 주파수 대역 상대 파워 | delta/theta/alpha/beta/gamma relative band power |

- **교차검증**: 피험자 단위(subject-level) 5-fold cross-validation

## Results

VMCI–HC 분류 과제 (AUC, mean ± std)

| 채널 구성 | EEGNet | qEEG + Logistic Regression |
|---|---|---|
| 19채널 (전체) | 0.746 ± 0.090 | 0.689 ± 0.098 |
| 후두부 2채널 (O1, O2) | 0.738 ± 0.051 | 0.656 ± 0.067 |

본 연구는 VMCI–HC 분류과제에서 최소 채널 EEG 기반 분류 가능성을 보여준다. 특히 연구에서 활용된
다양한 분류 모델들(EEGNet, qEEG 기반 로지스틱 회귀)은 후두부 2채널만으로도 19채널에 준하는 분류
성능을 보고함으로써, 최소 채널의 강건성을 시사한다. 향후, 추가적인 독립 코호트에서 최소 채널 측정
성능 평가가 필요하지만, 본 연구는 소수 채널을 활용한 임상 선별 보조 도구 개발 가능성을 가속화한다.

## Repository Structure

| 파일 | 설명 |
|---|---|
| `01_Vascular_mci_hc_vd.ipynb` | 메인 실험 파이프라인 — raw EEG 로딩, 채널/대역 ablation sweep, EEGNet/qEEG 학습·평가, VMCI–HC(본 초록 보고 결과) 및 VMCI–VD 분류 전체 수행 |
| `02_Qeeg_comparison_VMCI_VD.ipynb` | 사전 계산된 qEEG 밴드파워 CSV 기반 VMCI–VD 이진분류 경량 재검증 |
| `03_Qeeg_visualize_3group.py` | HC / MCI_vascular / SIVD 3그룹 qEEG 지표 비교 시각화(박스플롯 + Mann-Whitney U) |

파일명 앞 번호(01/02/03)는 실행/서사 순서를 나타냅니다: 메인 파이프라인 → qEEG 경량 재검증 → 3그룹 시각화.

## Keywords

Vascular Mild Cognitive Impairment, Electroencephalography, Deep Learning, Channel Selection,
Quantitative EEG

## References

[1] Kim, M., et al. "Deep Learning-Based EEG Analysis to Classify Normal, Mild Cognitive Impairment,
and Dementia." *NeuroImage*, vol. 272, 2023, 120054.
https://doi.org/10.1016/j.neuroimage.2023.120054

[2] Lawhern, V. J., et al. "EEGNet: a compact convolutional neural network for EEG-based
brain–computer interfaces." *J. Neural Eng.*, vol. 15, no. 5, 2018, 056013.
https://doi.org/10.1088/1741-2552/aace8c

## Acknowledgements

본 연구는 보건복지부 재원 및 한국보건산업진흥원의 지원(과제번호: No. RS-2025-02313201)과 보건복지부의
재원으로 한국보건산업진흥원의 보건의료기술 연구개발사업의 지원(No. RS-2026-25532408), 그리고
과학기술정보통신의 재원으로 한국연구재단(No. RS-2022-NR070859)의 지원을 받아 수행된 연구임.
