'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        # 부모 클래스인 nn.Module 초기화
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        
        # 임베딩 크기 (feature vector 차원 수)
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        
        # Dropout 비율 설정 (Node Dropout: 그래프 구조 드롭아웃, Message Dropout: 임베딩 값 드롭아웃)
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        # self-loop 포함된 인접행렬
        self.norm_adj = norm_adj
        
        # GNN 레이어별 출력 차원 크기 리스트
        self.layers = eval(args.layer_size)
        
        # L2 Regularization 가중치 (과적합 방지)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # Xavier Initialization (Glorot Initialization) 사용
        # 가중치를 적절한 분포로 초기화하여 학습 안정을 도모함
        initializer = nn.init.xavier_uniform_

        # 사용자 임베딩과 아이템 임베딩을 저장하는 딕셔너리
        embedding_dict = nn.ParameterDict({
            # n_user x emb_size 크기의 임베딩 행렬 만든 후 initialize로 가중치 초기화하여 nn.Parameter로 저장
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        # 레이어의 가중치(W)와 편향(b)을 저장하는 딕셔너리
        weight_dict = nn.ParameterDict()
        # layers에 입력값인 emb_size와 출력값인 self.layers를 연결하여 리스트 생성
        layers = [self.emb_size] + self.layers
        
        # 각 레이어마다 가중치 행렬 생성
        for k in range(len(self.layers)):
            # GC(Graph Convolution) 가중치: 이웃 정보를 합칠 때 사용
            # W_gc_%d 이름으로 입력인 layers[k]와 출력인 layers[k+1] 크기의 가중치 행렬 만든 후 initialize로 가중치 초기화하여 nn.Parameter로 저장
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            # b_gc_%d 이름으로 입력인 layers[k+1] 크기의 편향 벡터 만든 후 initialize로 가중치 초기화하여 nn.Parameter로 저장
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            # Bi-interaction(상호작용) 가중치: 자신과 이웃의 element-wise product 정보 변환에 사용
            # W_bi_%d 이름으로 입력인 layers[k]와 출력인 layers[k+1] 크기의 가중치 행렬 만든 후 initialize로 가중치 초기화하여 nn.Parameter로 저장
            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            # b_bi_%d 이름으로 입력인 layers[k+1] 크기의 편향 벡터 만든 후 initialize로 가중치 초기화하여 nn.Parameter로 저장
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    # sparse matrix를 sparse tensor로 변환
    def _convert_sp_mat_to_sp_tensor(self, X):
        # X는 self-loop 포함된 인접행렬
        # X를 COO format으로 변환
        coo = X.tocoo()
        # COO format의 row와 col 인덱스를 PyTorch LongTensor로 변환
        i = torch.LongTensor([coo.row, coo.col])
        # COO format의 data를 PyTorch FloatTensor로 변환
        v = torch.from_numpy(coo.data).float()
        # i, v, coo.shape를 사용하여 PyTorch sparse tensor 생성
        return torch.sparse.FloatTensor(i, v, coo.shape)

    # sparse tensor에 dropout 적용
    def sparse_dropout(self, x, rate, noise_shape):
        ''' ex) (dropout 비율)rate = 0.3, (만들 랜덤숫자 개수)noise_shape = 5 '''
        # dropout 확률 계산  ex) random_tensor = 1 - 0.3 = 0.7
        random_tensor = 1 - rate
        # random_tensor에 noise_shape만큼의 난수 추가 
        ''' ex) torch.rand(5) = [0.234, 0.876, 0.123, 0.456, 0.789]'''
        ''' ex) random_tensor += torch.rand(5) = [0.934, 1.576, 0.823, 1.156, 1.489]'''
        random_tensor += torch.rand(noise_shape).to(x.device)
        # random_tensor를 floor(소수점제거)하여 dropout mask 생성
        ''' ex) dropout_mask = [0, 1, 0, 1, 1] -> [False, True, False, True, True]'''
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        # x.indices()는 x의 row, col 인덱스를 반환
        i = x._indices()
        # x.values()는 x의 value를 반환
        v = x._values()

        # dropout mask를 적용하여 x의 row, col 인덱스와 value를 필터링
        '''
        ex)
            원본
            i = [[0,  0,  0,  1,  1],
                [12, 45, 67, 3, 89]]
            v = [1.0, 1.0, 1.0, 1.0, 1.0]

            마스크 적용 (True인 것만 남김)
            i = [[0,  0,  1],     # 2번째, 4번째, 5번째만
                [45, 3, 89]]
            v = [1.0, 1.0, 1.0]
        '''
        i = i[:, dropout_mask]
        v = v[dropout_mask]

        # dropout된 sparse tensor 생성
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        # 스케일 조정 후 반환
        return out * (1. / (1 - rate))

    # Bayesian Personalized Ranking (BPR -> 빈도주의) loss 계산
    def create_bpr_loss(self, users, pos_items, neg_items):
        '''
        ex)
        # 배치 크기 4
        users = [
            [0.1, 0.2, 0.3, ...],  # 사용자0 임베딩 (임베딩 64차원)
            [0.5, 0.1, 0.7, ...],  # 사용자1 임베딩
            [0.3, 0.8, 0.2, ...],  # 사용자2 임베딩
            [0.6, 0.4, 0.5, ...]   # 사용자3 임베딩
        ]  # shape: (4, 64)

        pos_items = [
            [0.8, 0.2, 0.1, ...],  # 긍정 아이템0 임베딩
            [0.3, 0.9, 0.4, ...],  # 긍정 아이템1 임베딩
            ...
        ]  # shape: (4, 64)

        neg_items = [
            [0.1, 0.5, 0.6, ...],  # 부정 아이템0 임베딩
            [0.7, 0.2, 0.3, ...],  # 부정 아이템1 임베딩
            ...
        ]  # shape: (4, 64)
        '''
        # 사용자와 긍정 아이템의 내적곱 계산
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        # 사용자와 부정 아이템의 내적곱 계산
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        # LogSigmoid 함수를 사용하여 BPR loss 계산
        '''
        ex)
        pos_scores = [0.85, 0.92, 0.73, 0.88]
        neg_scores = [0.35, 0.21, 0.48, 0.19]
                -  -----------------------
        차이       = [0.50, 0.71, 0.25, 0.69]

        # Sigmoid: 0~1 사이로 압축
        # LogSigmoid: log(sigmoid(x))

        차이가 크면 → 0에 가까움 (좋음, 손실 작음)
        차이가 작거나 음수 → 큰 음수 (나쁨, 손실 큼)

        maxi = [-0.47, -0.24, -0.92, -0.26]

        평균 = (-0.47 - 0.24 - 0.92 - 0.26) / 4 ≈ -0.47

        mf_loss = -1 * (-0.47) = 0.47 -> 손실은 양수여야하고, 작을수록 좋음
        '''
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        # BPR 손실 계산
        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer 정규화 진행하여 임베딩 값이 너무 커지지 않도록 제어
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        # 임베딩 손실 계산
        emb_loss = self.decay * regularizer / self.batch_size

        # 최종 손실 반환
        return mf_loss + emb_loss, mf_loss, emb_loss

    # 사용자와 아이템의 내적곱 계산
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        # 내적곱 계산을 위해 pos_i_g_embeddings를 전치
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        
        # 노드 드롭아웃 적용 (Graph Structure Learning 효과)
        # 인접 행렬의 일부 연결을 무작위로 제거하여 과적합 방지
        # drop_flag가 True일 때만 노드 드롭아웃 적용 -> 학습 시에만 적용하고 테스트 시에는 적용하지 않음
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        # 초기 임베딩 (Layer 0) - 사용자와 아이템 임베딩을 결합
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        # 모든 레이어의 임베딩을 저장할 리스트 (최종적으로 Concat 하기 위함)
        all_embeddings = [ego_embeddings]

        # GNN 레이어 통과 (Message Passing)
        for k in range(len(self.layers)):
            # 1. 인접 행렬과 현재 임베딩을 곱하여 이웃 노드의 정보를 가져옴
            # side_embeddings: 이웃 노드들의 임베딩 가중합
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # 2. Sum Aggregation (단순 이웃 정보 합)
            # W1 * (e_u + e_i) 형태의 변환
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # 3. Bi-Interaction Aggregation (상호작용 정보 합)
            # 3-1. Element-wise Product: 자신(ego)과 이웃(side)의 요소를 곱함 -> 특징 간 상호작용 포착
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # 3-2. W2 * (e_u . e_i) 변환
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            # 4. 활성화 함수 (LeakyReLU) 적용
            # Sum 정보와 Bi 정보(상호작용)를 합쳐서 비선형 변환
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)

            # 5. 메시지 드롭아웃 (Embedding Dropout) 적용
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # 6. 정규화 (임베딩의 크기를 1로 맞춤 - 학습 안정성)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            # 현재 레이어의 결과 임베딩을 리스트에 저장
            all_embeddings += [norm_embeddings]

        # 모든 레이어의 임베딩을 결합 (Concatenation)
        # High-order Connectivity 정보를 모두 활용하기 위함
        all_embeddings = torch.cat(all_embeddings, 1)
        
        # 결합된 전체 임베딩을 다시 사용자와 아이템 부분으로 분리
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
