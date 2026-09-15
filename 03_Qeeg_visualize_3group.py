"""
qEEG 지표 3그룹(HC / MCI_vascular / SIVD) 비교 시각화
=====================================================

혈관성 인지저하의 임상적 진행 단계(정상 -> 경도인지장애 -> 치매)에 따라 qEEG 지표가
어떻게 변하는지 박스플롯 + 개별점 + 통계적 유의성으로 시각화하는 스크립트.
try1.ipynb / Vascular_mci_hc_vd.ipynb의 분류(classification) 파이프라인과는 별개로,
분류에 쓰인 것과 같은 qEEG 지표를 그룹 간 비교/기술통계용으로 따로 시각화하는 용도.

입력 데이터
-----------
- SIVD_vascMCI_QEEG.csv : SIVD + MCI_vascular 피험자 132명의 qEEG 지표
  (Qeeg_comparison_VMCI_VD.ipynb와 동일 소스)
- QEEG_HC.xlsx : HC(정상대조군) 246명. 컬럼명이 "All_Channels_RelPow_Delta | Normal"
  처럼 그룹명이 붙어있는 포맷이라, " | Normal" 접미사를 잘라내고 컬럼명을 나머지 두
  그룹과 통일시킨 뒤 병합.

지표 (9개)
----------
- 상대 밴드파워 5개: Delta / Theta / Alpha / Beta / Gamma
- 비율 지표 4개: DAR(delta/alpha), TAR(theta/alpha), DTABR((delta+theta)/(alpha+beta))
  는 원본 CSV/XLSX에 이미 있는 값 그대로 사용. TBR(theta/beta)만 원본에 없어서
  Theta/Beta 상대파워로 직접 계산해서 추가(라벨에 TBR*로 표시해 구분).

주의: 표본 크기가 분류 파이프라인과 다름
----------------------------------------
Vascular_mci_hc_vd.ipynb의 분류 실험(EEGNet/qEEG ablation)은 HC 246명 중 70명만
랜덤 서브샘플링해서 썼지만, 이 시각화·통계비교 스크립트는 HC 246명 전체를 사용한다
(아래 print문으로도 명시). 즉 이 스크립트의 그림/표는 "정상군의 실제 분포를 보여주기
위한 기술통계"이고, 분류 성능 수치와는 표본이 다르므로 논문에서 같이 인용할 때는
"분류에는 70명 서브샘플, 그룹 비교 시각화에는 246명 전체 사용"이라고 구분해서 밝힐 것
— 안 그러면 두 결과의 n이 안 맞아 보여서 리뷰어가 의아해할 수 있음.

통계 로직
---------
- 세 그룹의 임상적 진행 순서(GROUP_ORDER = [HC, MCI_vascular, SIVD])를 x축 순서로 고정
- 비교 쌍 3개: HC-MCI_vascular(인접 단계), MCI_vascular-SIVD(인접 단계),
  HC-SIVD(전체 진행 비교)
- 검정: 정규성을 가정하지 않는 Mann-Whitney U(비모수) — qEEG 비율 지표처럼 분포가
  치우치기 쉬운 데이터에 적합한 선택
- 유의수준 표시: * p<0.05, ** p<0.01, *** p<0.001, 그 외 n.s.
- draw_bracket(): 두 박스 사이에 "ㄷ"자 모양 브라켓을 그리고 그 위에 유의성 기호를
  얹는 커스텀 함수. 비율 지표(log scale) / 밴드파워(선형 scale) 두 경우 모두 브라켓
  높이를 겹치지 않게 자동 계산.

  참고(방법론): 9개 지표 x 3쌍 = 총 27번의 가설검정을 다중비교 보정(Bonferroni/FDR
  등) 없이 그대로 표시하고 있음. 탐색적 시각화용으로는 괜찮지만, 이 그림의 유의성
  표시를 논문 본문에서 직접 인용할 거라면 다중비교 보정 여부를 한 번 검토해볼 것.

출력 3가지
----------
1. qeeg_3group_comparison.png       — 9개 지표(2x5 그리드, 마지막 칸은 비움)
                                       박스플롯+산점도+브라켓
2. qeeg_3group_spectral_profile.png — 그룹별 평균 밴드파워 막대그래프(5개 밴드, ±SEM)
3. 콘솔 출력                          — 지표별 그룹 평균±표준편차 + 3쌍 p-value 요약표

색상 / 스타일
-------------
HC=#1baf7a(초록), MCI_vascular=#2a78d6(파랑), SIVD=#eb6834(주황) — 파랑/주황은 다른
결과(sensitivity/specificity 그래프)에서 쓴 것과 같은 팔레트를 그대로 이어서 씀
→ 여러 그림을 나란히 놓아도 색상 의미가 일관됨(파랑=MCI_vascular, 주황=SIVD).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import mannwhitneyu

# ---------------- 한글 폰트 설정 ----------------
for cand in ["Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 설정 ----------------
CSV_PATH = "C:/eeg_research/Vascular_Subset/SIVD_vascMCI_QEEG.csv"    # SIVD, MCI_vascular
HC_XLSX_PATH = "C:/eeg_research/HC_Subset/QEEG_HC.xlsx"               # HC (정상대조군)

BANDPOWER_COLS = [
    "All_Channels_RelPow_Delta", "All_Channels_RelPow_Theta", "All_Channels_RelPow_Alpha",
    "All_Channels_RelPow_Beta", "All_Channels_RelPow_Gamma",
]
RATIO_COLS = ["All_Channels_Ratio_DAR", "All_Channels_Ratio_TAR", "All_Channels_Ratio_DTABR", "All_Channels_Ratio_TBR"]
ALL_COLS = BANDPOWER_COLS + RATIO_COLS

LABELS = {
    "All_Channels_RelPow_Delta": "Delta\n(0.5–4Hz)",
    "All_Channels_RelPow_Theta": "Theta\n(4–8Hz)",
    "All_Channels_RelPow_Alpha": "Alpha\n(8–13Hz)",
    "All_Channels_RelPow_Beta": "Beta\n(13–30Hz)",
    "All_Channels_RelPow_Gamma": "Gamma\n(30–45Hz)",
    "All_Channels_Ratio_DAR": "DAR\n(δ/α)",
    "All_Channels_Ratio_TAR": "TAR\n(θ/α)",
    "All_Channels_Ratio_DTABR": "DTABR\n((δ+θ)/(α+β))",
    "All_Channels_Ratio_TBR": "TBR*\n(θ/β, 계산값)",
}

# 정상 -> 경도인지장애 -> 치매로 이어지는 임상적 진행 순서로 표시
GROUP_ORDER = ["HC", "MCI_vascular", "SIVD"]
COLORS = {"HC": "#1baf7a", "MCI_vascular": "#2a78d6", "SIVD": "#eb6834"}

# 표시할 쌍대비교 (인접 진행단계 2쌍 + HC-SIVD 전체비교 1쌍)
PAIRS = [("HC", "MCI_vascular"), ("MCI_vascular", "SIVD"), ("HC", "SIVD")]

# ---------------- 로드: SIVD / MCI_vascular ----------------
df_vasc = pd.read_csv(CSV_PATH)
df_vasc["All_Channels_Ratio_TBR"] = df_vasc["All_Channels_RelPow_Theta"] / df_vasc["All_Channels_RelPow_Beta"]
df_vasc = df_vasc[["Group"] + ALL_COLS]

# ---------------- 로드: HC ----------------
# 컬럼명이 "All_Channels_RelPow_Delta | Normal" 식으로 돼있어서 " | Normal" 접미사 제거
hc_raw = pd.read_excel(HC_XLSX_PATH, sheet_name=0)
hc_raw.columns = [c.split(" | ")[0].strip() for c in hc_raw.columns]
hc_raw = hc_raw.rename(columns={hc_raw.columns[0]: "Group"})
hc_raw["Group"] = "HC"
hc_raw["All_Channels_Ratio_TBR"] = hc_raw["All_Channels_RelPow_Theta"] / hc_raw["All_Channels_RelPow_Beta"]
df_hc = hc_raw[["Group"] + ALL_COLS]

# ---------------- 3그룹 합치기 ----------------
df = pd.concat([df_hc, df_vasc], ignore_index=True)
counts = df["Group"].value_counts()
print(f"총 {len(df)}명 | " + ", ".join(f"{g}={counts.get(g, 0)}" for g in GROUP_ORDER))
print("(참고: 분류기 학습 때는 HC 246명 중 70명만 랜덤 서브샘플링해서 썼지만, "
      "이 시각화·통계비교에서는 정상군 분포를 더 정확히 보여주기 위해 246명 전체를 사용)")


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def draw_bracket(ax, x1, x2, y, gap, text, log_scale):
    """두 박스 사이에 유의성 브라켓(ㄷ자 모양 선 + 텍스트)을 그림."""
    if log_scale:
        y_top = y * gap
        ax.plot([x1, x1, x2, x2], [y, y_top, y_top, y], color="#52514e", lw=1)
        ax.text((x1 + x2) / 2, y_top * 1.03, text, ha="center", va="bottom", fontsize=8)
        return y_top
    else:
        y_top = y + gap
        ax.plot([x1, x1, x2, x2], [y, y_top, y_top, y], color="#52514e", lw=1)
        ax.text((x1 + x2) / 2, y_top + gap * 0.06, text, ha="center", va="bottom", fontsize=8)
        return y_top


# ---------------- 그림 1: 9개 지표, 3그룹 비교 (박스플롯 + 개별점 + 3쌍 Mann-Whitney U) ----------------
fig, axes = plt.subplots(2, 5, figsize=(21, 9))
axes = axes.flat
rng = np.random.default_rng(0)

for ax, col in zip(axes, ALL_COLS):
    log_scale = col in RATIO_COLS
    data_by_group = [df.loc[df["Group"] == g, col].dropna().values for g in GROUP_ORDER]

    bp = ax.boxplot(data_by_group, tick_labels=GROUP_ORDER, showfliers=False, widths=0.5, patch_artist=True)
    for patch, g in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.35)
    for median in bp["medians"]:
        median.set_color("black")

    for i, (g, vals) in enumerate(zip(GROUP_ORDER, data_by_group), start=1):
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=COLORS[g], s=10, alpha=0.5, edgecolors="none")

    if log_scale:
        ax.set_yscale("log")

    # whisker 최댓값(이상치 제외) 기준으로 브라켓 시작 높이 잡기
    whisker_max = max(np.percentile(v, 75) + 1.5 * (np.percentile(v, 75) - np.percentile(v, 25)) for v in data_by_group)
    whisker_max = max(whisker_max, max(v.max() for v in data_by_group) * (0.6 if log_scale else 1.0))
    y0 = whisker_max

    # (1) 인접 단계 두 쌍: HC-MCI, MCI-SIVD — 낮은 높이
    gap1 = (y0 * 1.15) if log_scale else (y0 * 0.12 + 1e-9)
    for (g1, g2), (x1, x2) in zip(PAIRS[:2], [(1, 2), (2, 3)]):
        v1 = df.loc[df["Group"] == g1, col].dropna().values
        v2 = df.loc[df["Group"] == g2, col].dropna().values
        _, p = mannwhitneyu(v1, v2, alternative="two-sided")
        draw_bracket(ax, x1, x2, y0, gap1, sig_label(p), log_scale)

    # (2) HC vs SIVD (전체 진행단계) — 더 높은 높이
    y1 = y0 * gap1 * 1.25 if log_scale else y0 + gap1 * 1.6
    gap2 = gap1 * 1.3 if log_scale else gap1 * 1.3
    v1 = df.loc[df["Group"] == "HC", col].dropna().values
    v2 = df.loc[df["Group"] == "SIVD", col].dropna().values
    _, p = mannwhitneyu(v1, v2, alternative="two-sided")
    draw_bracket(ax, 1, 3, y1, gap2, sig_label(p), log_scale)

    ax.set_title(LABELS[col], fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
    if log_scale:
        ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    else:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)

axes[-1].axis("off")
# axes[-1].text(
#     0.0, 0.9,
#     "그룹: HC(n={})  /  MCI_vascular(n={})  /  SIVD(n={})\n\n".format(
#         counts.get("HC", 0), counts.get("MCI_vascular", 0), counts.get("SIVD", 0))
#     + "박스플롯: 중앙값·IQR (이상치 제외 표시)\n점: 피험자 1명당 1개\n\n"
#     + "브라켓 3개 = HC↔MCI, MCI↔SIVD, HC↔SIVD\n(Mann-Whitney U, 비모수)\n"
#     + "유의수준: * p<0.05  ** p<0.01  *** p<0.001\n\n"
#     + "* TBR은 원본 데이터에 없어\nTheta/Beta 상대파워로 직접 계산\n\n"
#     + "HC는 학습때 쓴 70명 서브샘플이 아니라\n전체 246명 기준",
#     fontsize=9.5, va="top", transform=axes[-1].transAxes,
# )

fig.suptitle("HC vs MCI_vascular vs SIVD — qEEG 지표 3그룹 비교", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("qeeg_3group_comparison.png", dpi=150, bbox_inches="tight")
print("저장: qeeg_3group_comparison.png")

# ---------------- 그림 2: 그룹별 평균 스펙트럼 프로파일 (밴드파워 5개, 3그룹) ----------------
fig2, ax2 = plt.subplots(figsize=(9, 5))
means = df.groupby("Group")[BANDPOWER_COLS].mean()
sems = df.groupby("Group")[BANDPOWER_COLS].sem()
x = np.arange(len(BANDPOWER_COLS))
width = 0.26
for i, g in enumerate(GROUP_ORDER):
    ax2.bar(
        x + (i - 1) * width, means.loc[g, BANDPOWER_COLS].values, width,
        yerr=sems.loc[g, BANDPOWER_COLS].values, capsize=3,
        label=f"{g} (n={counts.get(g, 0)})", color=COLORS[g], alpha=0.85,
    )
ax2.set_xticks(x)
ax2.set_xticklabels(["Delta", "Theta", "Alpha", "Beta", "Gamma"])
ax2.set_ylabel("평균 상대밴드파워 (± SEM)")
ax2.set_title("그룹별 평균 스펙트럼 프로파일 (Relative Band Power) — HC / MCI_vascular / SIVD")
ax2.legend()
fig2.tight_layout()
fig2.savefig("qeeg_3group_spectral_profile.png", dpi=150, bbox_inches="tight")
print("저장: qeeg_3group_spectral_profile.png")

# ---------------- 콘솔 요약표: 3그룹 평균±표준편차 + 3쌍 p-value ----------------
print("\n=== 그룹별 평균±표준편차, 쌍대비교 p-value ===")
header = f"{'지표':32s} {'HC':>16s} {'MCI_vascular':>16s} {'SIVD':>16s}   {'HC-MCI':>9s} {'MCI-SIVD':>9s} {'HC-SIVD':>9s}"
print(header)
for col in ALL_COLS:
    means_row = df.groupby("Group")[col].mean()
    stds_row = df.groupby("Group")[col].std()
    cells = "  ".join(f"{means_row[g]:.3f}±{stds_row[g]:.3f}" for g in GROUP_ORDER)

    pvals = []
    for g1, g2 in PAIRS:
        v1 = df.loc[df["Group"] == g1, col].dropna().values
        v2 = df.loc[df["Group"] == g2, col].dropna().values
        _, p = mannwhitneyu(v1, v2, alternative="two-sided")
        pvals.append(f"{p:.3g}{sig_label(p) if sig_label(p)!='n.s.' else ''}")
    print(f"{col:32s} {cells}   " + "  ".join(f"{p:>9s}" for p in pvals))
