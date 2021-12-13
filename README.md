# p3-dst-dts
- 기간 : 2021.04.26~2021.05.21
- 대회 내용 : 다중 도메인(관광/식당/숙소/택시/지하철) 사용자-시스템 대화 데이터를 학습하여 사용자가 원하는 바를 state으로 뽑아내는 과제 (JGA 0.6423, 최종 8등 팀)

![image](https://user-images.githubusercontent.com/52443401/145763906-75399b3c-22e3-4e42-8476-55d3717e2aa8.png)
- 수행 요약 : EDA, validation dataset 오류 분석을 통한 방향성 제시, BART를 이용한 CoCo(Controllable Counterfactuals) data augmentation


### 사용한 모델
- ontology 기반의 SUMBT

![image](https://user-images.githubusercontent.com/52443401/145764250-0fce1a07-d9c9-4941-ba2a-5c0b60d178b0.png)

- open vocab 기반 TRADE
![image](https://user-images.githubusercontent.com/52443401/145764404-407210b0-c1ff-483c-8fe1-bf449991ea1e.png)

### 오류 분석 및 방향성
- 택시, 지하철, 식당 도메인에서 "시간"과 관련된 slot을 만날 때 두 모델 모두 어려워했다.
![slot_percent_barplot_ep20](https://user-images.githubusercontent.com/52443401/145764775-ecb58c8d-d71b-4760-ba29-d9b8dc70b4ed.png)

- 데이터 내 시간 표현의 다양성 발견 
- "13:30" -> ["오후 1시 30분", "1시 반", "낮 1시 30분", "13:30" ....etc]
- 자연스럽게 하나의 시간표현을 여러 시간표현으로 만들면서 문맥을 해치지 않는 augmentation 필요

### BART & CoCo(Controllable Counterfactuals) 을 통한 data augmentation
- 여러 개의 turn 들로 구성된 하나의 dialogue 내 시간표현들을 다양하게 만들어주면서 전체 맥락을 해치지 않는 augmentation 방법 고안
- 마지막 user 발화만 기존 slot-state 1개를 제거하고 새로운 state을 추가하여 새로운 문장을 생성, 기존 상위 turn의 text들은 BART를 거쳐 문맥상의 변화만 주고 state는 그대로 표현(시간 표현이 있다면 확률적으로 다른 시간 표현으로 변경함)
![image](https://user-images.githubusercontent.com/52443401/145765432-497eae92-dd07-40e6-8d6c-358f358b8d5e.png)

