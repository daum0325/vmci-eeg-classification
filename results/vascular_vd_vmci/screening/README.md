# 스크리닝 결과 (3-fold × 1-repeat × 20-epoch)

이 폴더의 JSON들은 `ablation_summary_vascular.json` 및 11개 조합 개별 결과로, 채널/주파수 대역
11개 조합 중 어떤 조합을 5-fold × 5-repeat 풀세팅으로 재확인할지 추리기 위한 **빠른 스크리닝
단계 결과**입니다.

**최종 보고 수치가 아닙니다.** 실제로 재확인된 최종 수치는 `../full/` 폴더를 참고하세요
(`central_broadband`, `all_theta` 두 조합만 5-fold × 5-repeat로 재확인됨).

- `ablation_eegnet_all_broadband.json`은 별도 실행(run_tag 미지정) 도중 같은 파일 경로가
  덮어써지면서 원본 스크리닝 결과의 raw fold 데이터가 유실되어 이 폴더에는 포함하지 않았습니다.
  해당 조합의 집계 수치(mean/std)는 `ablation_summary_vascular.json` 안에 보존되어 있습니다.
