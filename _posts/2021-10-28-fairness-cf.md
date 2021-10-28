---
title:  "[Paper Review] Beyond Parity: Fairness Objectives for Collaborative Filtering"
toc: true
categories:
  - Study
  - Paper
  - Recommendations
tags:
  - Review 
---


> Beyond Parity: Fairness Objectives for Collaborative Filtering (NIPS '17) 논문 리뷰입니다.

## Introuduction

- recommendation에서 unfairness 문제 formalize
- demographic parity의 insufficiency
- **4가지 unfairness metrics** for recommendations 제안
- **5가지 fairness objectives** 제안
    - unfairness 패널티를 regularizer로 추가함
- 모델은 matric factorization만 사용해서 실험함

---

## **Related Work**

- sensitive features(e.g. gender, race, or age)를 지우는건 unfairness 개선에 도움이 되지만, 주로 지우는 것만으로는 부족함 - Pre-processing
    - unprotected features(↔ sensitive features)가 사실상 sensitive features와 관련될 수 있기 때문임
- 그래서 Demographic parity 또는 Equal Opportunity 등을 기반으로 하는 패널티를 regurlaizer로 사용하여 fairness 개선하는 연구들 등장 - In-processing

**Demographic Parity**

$Pr\{Y_{pred} =1|A=0\}=Pr\{Y_{pred} =1|A=1\}$
				
			
		

- group간의 statistical parity를 달성하는 것이 목적임
- protected group(=sensitive group)의 모든 members가 positive일 확률이 unprotected group의 positive일 확률과 같도록 하는 것
- 단, Demographic parity는 선호도가 sensitive features와 unrelated일 때만 적용할 수 있음
    - but, 유저 선호도는 보통 sensitive features(e.g. age, gender, race)에 의해 영향을 받음
- 따라서 추천에 적합한 방법은 아니다.

**Equal Opportunity**

$Pr\{Y_{pred} =1|A=0, Y=y\}=Pr\{Y_{pred} =1|A=1, Y=y\}, y=\{0, 1\}$
				
			
		

- sensitive group의 true positive rate과 unprotected group의 true positive rate이 같도록 하는 것
- Demographic 단점 해결
    - sensitive feature가 유저 선호도와 관련되어 있을 때도 적용 가능 → 추천에 적합한 방법

---

## **Fairness Objectives for Collaborative Filtering**

### 1. Matirx Factorization for Recommendation

![스크린샷 2021-10-27 오후 8.36.27.png](/assets/posts/스크린샷_2021-10-27_오후_8.36.27.png)

### 2. Unfair Recommendations from Underrepresentation

- `population imblance`, `observation bias` 개선에 focus

**Population Imbalance**

- STEM 예시에서는 사회적 unfairness 등으로 STEM에 success하지 못한 여성보다 success한 여성이 거의 없는 경우 (반대로 남성은 success한 경우가 더 많은 경우) → 애초에 data imbalance 문제 야기

**Observation Bias**

- data imbalance 문제랑은 상관 없음
- feedback loop로 발생하는 bias
    - 사용자에게 계속 추천되는 item은 계속 추천되고, 한번도 추천되지 못한 아이템은 계속 feedback 못받으니까 추천 안되는 현상
    - e.g. STEM course에서 여성이 training data에 많이 없으니까 여성에게는 STEM course가 추천되지 않는 것. feedback loop를 통해 fairnes gap이 더 커짐

### 3. Fairness Metrics

![스크린샷 2021-10-27 오후 11.37.35.png](/assets/posts/스크린샷_2021-10-27_오후_11.37.35.png)

- 아래 5개 unfairness metric을 regularizer로 사용함

**value unfairness**

- 각 group간 error(average predicted score와 average ratings간 error) difference를 unfairness로 정의
- e.g. female group과 male group의 error 차이가 클수록 unfairness

![스크린샷 2021-10-27 오후 11.18.40.png](/assets/posts/스크린샷_2021-10-27_오후_11.18.40.png)

**absolute unfairness**

![스크린샷 2021-10-27 오후 11.18.48.png](/assets/posts/스크린샷_2021-10-27_오후_11.18.48.png)

****

**underestimation unfairness**

![스크린샷 2021-10-27 오후 11.18.54.png](/assets/posts/스크린샷_2021-10-27_오후_11.18.54.png)

****

**overestimation unfairness**

![스크린샷 2021-10-27 오후 11.18.58.png](/assets/posts/스크린샷_2021-10-27_오후_11.18.58.png)

****

**non-parity unfairness**

![스크린샷 2021-10-27 오후 11.19.03.png](/assets/posts/스크린샷_2021-10-27_오후_11.19.03.png)

****