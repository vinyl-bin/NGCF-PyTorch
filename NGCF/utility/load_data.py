'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):

    # 데이터 읽고 희소행렬로 저장
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items 변수 초기화
        self.n_users, self.n_items = 0, 0   # 사용자 수, 아이템 수
        self.n_train, self.n_test = 0, 0    # 학습 아이템 데이터 수, 테스트 아이템 데이터 수
        self.neg_pools = {}                 # 부정 ??
        self.exist_users = []               # 존재하는 유저아이디 리스트

        # 학습 데이터 처리
        with open(train_file) as f:
            for l in f.readlines():  # 한 줄씩 처리
                if len(l) > 0:       # 공백이 아닐 때 
                    l = l.strip('\n').split(' ') # 양 끝 개행문자 삭제, 공백을 기준으로 쪼개서 리스트로
                    items = [int(i) for i in l[1:]] # 아이템은 인덱스 1부터
                    uid = int(l[0])                 # 유저아이디는 인덱스 0
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items)) # 기존 아이템 수와 현재 아이템 수 더 큰 걸 저장
                    self.n_users = max(self.n_users, uid) # 기존 유저수와 현재 유저 수 중 더 큰 걸 저장 
                    self.n_train += len(items) # 학습 아이템수만 누적

        # 테스트 데이터 처리  -> 왜 테스트에서는 max uid를 구하지 않는지 -> 테스트와 학습 데이터 uid 개수가 동일함
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n') # 양 끝 개행문자 삭제
                    try:
                        items = [int(i) for i in l.split(' ')[1:]] # 공백을 기준으로 쪼갠 후 리스트로 
                    except Exception:   # 모든 오류 무시 후 continue 처리
                        continue
                    self.n_items = max(self.n_items, max(items))  # 기존 아이템 수와 현재 아이템 수 더 큰 걸 저장
                    self.n_test += len(items) # 테스트 아이템수만 누적
        self.n_items += 1  # 아이템 id는 0부터 시작하므로 max 아이템 id에서 1을 더해야 실제 전체 아이템 id의 개수가 나옴
        self.n_users += 1  # uid가 0부터 시작하므로 max uid에서 1를 더해야 실제 전체 uid 개수가 나옴

        # 위에서 구한 변수들의 통계값 출력
        self.print_statistics()

        # 빈 희소행렬 (n_users X n_items) 크기로 생성 -> 아마도 train용 희소행렬로 예상됨
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        # uid인 key와 item인 value로 저장할 예정 
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    # uid 행과 item 열 부분에 1.0 표시
                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    # uid인 key와 item인 value로 저장
                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    # uid인 key와 item인 value로 저장
                    self.test_set[uid] = test_items

    # 캐싱된 인접행렬 가져오는 함수
    def get_adj_mat(self):

        # 이미 캐싱된 게 있을 때
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        # 만들어진게 없을 때 새로 인접 행렬 만들고 저장(캐싱된 게 없을 때)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    # 새로운 인접행렬(모든 사용자와 아이템, 사용자와 사용자, 아이템과 아이템의 관계) 만드는 함수
    def create_adj_mat(self):
        t1 = time()
        # 빈 희소행렬 (self.n_users + self.n_items X self.n_users + self.n_items) 크기로 생성
        # 전체 그래프의 노드 수 = 사용자 수 + 아이템 수
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        
        # 행렬 값 할당을 빠르게 하기 위해 LIL(List of Lists) 형식으로 변환
        adj_mat = adj_mat.tolil()
        
        # 사용자-아이템 상호작용 행렬 R도 LIL로 변환
        R = self.R.tolil()

        # 빈 adj_mat 희소행렬의 n_users들의 행으로만 이루어져 있고 n_items 열로만 이루어져 있는 부분을 R로 치환
        adj_mat[:self.n_users, self.n_users:] = R
        # adj_mat 희소행렬의 n_items들의 행으로만 이루어져 있고 n_users 열로만 이루어져 있는 부분을 R의 전치행렬로 치환
        adj_mat[self.n_users:, :self.n_users] = R.T
        # 인접 행렬 구성:
        # |   0   |   R   |
        # |-------|-------|
        # |  R^T  |   0   |
        

        # 다시 DOK 형식으로 변환
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        # 정규화 진행
        def mean_adj_single(adj):
            # 행 단위 정규화 (D^-1 * A)
            # rowsum: 각 행의 원소 합 (차수, Degree)
            rowsum = np.array(adj.sum(1))

            # 차수의 역수 계산 (1/Degree)
            d_inv = np.power(rowsum, -1).flatten()
            # 무한대 값(0으로 나눈 경우) 처리
            d_inv[np.isinf(d_inv)] = 0.
            # 대각 행렬 생성
            d_mat_inv = sp.diags(d_inv)

            # 정규화된 행렬 계산: D^-1 * A
            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        # 정규화 진행 시 단위행렬 더해서 주대각선이 1이 되도록 설정하여 self-loop 포함함
        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = mean_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            # 유저가 배치 크기보다 크므로 중복되지 않게 유저 샘플링
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            # 유저가 배치크기보다 작으므로 중복되게 유저 샘플링
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # num 크기만큼 user가 본 아이템을 고름
        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        # num 크기만큼 user가 보지 않은 아이템을 고름
        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                # 유저가 보지 않은 아이템이여야하고 배치 리스트에 중복되지 않은 아이템이어야 함
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            # 각 사용자에 대해 긍정 아이템 1개, 부정 아이템 1개 샘플링
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
