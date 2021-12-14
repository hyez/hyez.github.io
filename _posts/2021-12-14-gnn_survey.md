---
title:  "[Paper Review] Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions"
toc: true
categories:
  - Study
  - Paper
  - Recommendations
tags:
  - Review 
---

> Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions

## 1. Introduction

- **model-based CF methods**
    - complex user behaviour를 잡을 수 없음
    - e.g. matrix factorization(MF), factorization machine(FM)
- **neural network-based models**
    - high-order structural information을 잡을 수 없음 (user가 interact한 item만 포함함)
    - e.g. NCF, DeepFM
- **graph neural networks**
    - embedding propagation 적용
    - high-order neighors' information 접근 가능
    - **challenges**
        - graph constrcution → GNN의 input data는 어떻게 구성할 것인가?
        - network design → 설계 방법에 따라 장단점이 있음 (how to propagate and aggregate)
        - model optimization
        - computation cost

---

## 2. Background

### 2.1 Recommender Systems

- Overview
    
    ![스크린샷 2021-12-04 오후 10.22.42.png](/assets/posts/2021-12-14-gnn_survey/스크린샷_2021-12-04_오후_8.04.05.png)
    
- **Stages**
    
    ![스크린샷 2021-12-04 오후 8.04.05.png](/assets/posts/2021-12-14-gnn_survey/스크린샷_2021-12-04_오후_10.22.42.png)
    
    - Matching (candidate generation)
        - 수백만개 아이템 중에서 수백개 정도의 candidate 생성
    - Raking (CTR prediction)
        - scoring
    - Re-ranking (post-processing)
        - freshness, diversity, fairness 등을 위한 전략
    
- **Scenarios**
    - **Social Recommendation**
        - social relation(친구 가족 등)이 item preference에 영향을 미침
        
        ![Untitled](/assets/posts/2021-12-14-gnn_survey/Untitled.png)
        
    - **Sequential Recommendation**
        
        ![Untitled](/assets/posts/2021-12-14-gnn_survey/Untitled 1.png)
        
    - **Session-based Recommendation**
        - user profile이나 long-term historical interaction 데이터 사용이 불가능할때 short session만 사용하는 것
    - **Bundle Recommendation**
        - item 홍보, 마케팅 목적 → combination of items를 유저에게 추천함
        - e.g. Spotify, Pinterest, Amazon, IKEA 등에서 적용
        
        ![Untitled](/assets/posts/2021-12-14-gnn_survey/Untitled 2.png)
        
    - **Cross-Domain Recommendation (CDR)**
        - 유저는 multiple domain에서 multi-modal information으로 상호작용 함
            - e.g. 쿠팡에서 상품 샀다가 멜론에서 음악 들었다가 인스타로 이것저것 구경하고 이런거..
        - cold start, data sparsity 문제 해결하기 위함
    - **Multi-behavior Recommendation**
        - 유저는 mutiple types of behavior를 함
            - e.g. 비디오를 `click`, `collecting`, or `comment`를 달 수 있음
        - 이런 각각의 type을 할 확률을 예측하는 정확도 높이기가 목표

- **Objectives**
    - **Diversity**
        - **individual-level diversity**
            - 추천된 Item의 `dissimilarity` 측정
            - recommendation list가 얼마나 많은 topic을 cover하는지, 다양한 topic의 item들이 균형있게 추천되었는지 등을 말함
        - **system-level diversity**
            - low-level system diversity → 모든 유저에게 그냥 인기있는 아이템 추천함
            - 시스템 관점에서 아이템이 유저마다 다양하게 추천되었는지를 의미함
        
        ![Untitled](/assets/posts/2021-12-14-gnn_survey/Untitled 3.png)
        
    - **Explainability**
        - 왜 이 유저에게 이 아이템이 추천되었는지를 설명할 수 있는 것 (explainable)
        - transparent logic (↔ black box)
    - **Fairness**
        - 추천 시스템에선 "데이터" 혹은 "알고리즘"에 의해 bias가 생김 → fairness 다루는 것도 중요
        - `user fairness`
            - no algorithmic bias  among specific users or demographic groups
        - `item fairness`
            - 아이템들의 fair exposure를 의미
            - no popularity bias (인기있는 아이템들이 더 많이 추천되는거)
            - two methods to enhance fairness
                - debiasing recommendation results in training process
                - re-ranking (post-processing)
            - graph에서 fairness를 다루기 어려움
                - graph data(e.g. user-user)가 unfairness를 극대화함
                

### 2.2 Graph Neural Networks

![Untitled](/assets/posts/2021-12-14-gnn_survey/Untitled%204.png)

![Untitled](/assets/posts/2021-12-14-gnn_survey/Untitled%205.png)

- **Graph Construction**
    - `Homogeneous graph`
        - each edge connects only two nodes
        - only one type of nodes and edges
    - `Heterogeneous graph`
        - each edge connects only two nodes
        - multiple types of nodes or edges
    - `Hypergraph`
        - each edge joins more than two nodes.
- **Network Design [[링크]](https://distill.pub/2021/understanding-gnns/)**
    - `GCN`
        - combines graph convolution and neural networks
    - `GraphSAGE`
        - samples neighbors of the target node
        - aggregates their embeddings
            - function AGGREGATE has various options, such as `MEAN`, `LSTM`
        - merges with the target embedding to update
    - `GAT`
        - attention mechanisms to aggregate neighborhood features (embeddings)
    - `HetGNN`
        - heterogeneous graphs (multiple types of nodes and edges)
        - HetGNN first divides neighbors into subsets according to their types.
        - aggregator function is conducted for each type of neighbor to gather localized information
            - combining LSTM and MEAN operations
        - different types of neighborhood information are aggregated based on the attention mechanism
    - `HGNN`
        - hypergraph
        - two stages of propagating neighborhood embeddings
            - propagation from nodes to the hyperedge connecting them
            - propagation from hyperedges to the node they meet.
    
- **Model Optimization**
    - pair-wise
    - point-wise

### 2.3 Why are GNNs required for recommender systems

- **Structural data**
    - online data : `user-item interaction` (rating, click, purchase, etc.), `user profile` (gender, age, income, etc.), `item attribute` (brand, category, price, etc.)
    - *Traiditional recommender systems* → one or a few specific data만 다룸 → sub-optimal performance
    - *GNN* → strong power in learning representations, high-quality embeddings for users, and other features can be obtained
- **High-order connectivity**
    - *CF* → only contain directly connected item (only first-order connectivity)
    - *GNN* → multi-hop neighbors on the graph (high-order connectivity)
- **Supervision signal**
    - GNN based model은 supervised learning 추천에서 data sparsity 문제를 해결할 수 있음
    - *GNN* → **semi-supervised** signals in the representation learning process

---

## 3. Challenges of Applying GNNs to Recommender Systems

### 3.1 Graph Constrcution

- **Nodes (users/items)**
    - 다른 node type을 구분하기 어려움 (이게 item 노드인지 user 노드인지)
    - 구체적인 input(e.g. numerical features)을 다루기가 어려움
        - categorical feature로 매핑해서 해결하는 방법 등이 있음
- **Edges (interactions)**
    - edges의 정의가 graph의 quality에 영향을 끼침
        - edge가 user-item interaction인지, user-user relation인지 등
    - too-dense edge → a very large number of neighbors
        - solutions : sampling, filtering, or pruning on graphs
    - too-sparse edge → poor utility of embedding propagation
    

### 3.2 Network Design

- propagation layer를 어떻게 설계하는지도 중요함
    - path를 어떻게 정의할 것인지, parametric으로 할 것인지 등
        - e.g. item embeddings을 user node로 propagation → item-based CF
- various choices of aggregation functions
    - mean pooling, LSTM, max, min, etc.
- 이들은 computation efficiency에 영향을 끼침!

### 3.3 Model Optimization

- loss
    - logloss → point-wise link prediction loss
    - BPR loss → pair-wise link prediction loss
- data sampling
    - positive/negative sampling 방법 → graph structure에 의존적임
    - random walk 등이 있음

### 3.4 Computation Efficiency

- 특히 spectral GNN은 NCF, FM 등이랑 비교했을 때 computation cost가 훨씬 큼
- spatial GNN models (such as PinSage) → large-scale에 훨씬 더 적합함
    - sampling among neighbors
    - pruned graph structure

---

## References

- [https://arxiv.org/pdf/2109.12843.pdf](https://arxiv.org/pdf/2109.12843.pdf)