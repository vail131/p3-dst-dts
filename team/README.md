### 1. 파일 설명

#### code 폴더

`code`: 코드 있는 폴더

#### ipynb 파일들

`coco.ipynb`: coco aug에 사용된 파일, generate해서 under-generation 필터까지 하고 결과를 저장

`over_gen.ipynb`: coco.ipynb로부터 받은 데이터로 over-generation 필터 적용해서 결과 저장

`find_word.ipynb`: under-generation 필터 작성할때 사용한 파일

`분포.ipynb`: coco aug 레어한 케이스 찾을려고 만든 파일

`ensemble.ipynb`: hard voting 앙상블 파일, slot-domain별로 hard voting함

`merge_results.ipynb`: 특정 domain-slot을 다른 모델의 결과로 바꿀려고 할 때 사용

