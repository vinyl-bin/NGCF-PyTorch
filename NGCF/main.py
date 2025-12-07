'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import warnings
warnings.filterwarnings('ignore')
from time import time

# 파일 직접 실행해야 돌아가도록 설정
if __name__ == '__main__':

    # args 객체에서 device라는 새로운 변수 할당 및 초기화
    # 옵션에서 입력한 번호의 GPU 사용 
    args.device = torch.device('cuda:' + str(args.gpu_id))

    # 기본 인접행렬, 정규화된 인접행렬(self-loop 포함), 평균 인접행렬 캐싱된 걸 가져오거나 새로 만들기
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    # Dropout 비율 설정 (eval을 사용하여 문자열을 리스트 등으로 변환)
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)

    # 사용자수, 아이템수, 정규화된 인접행렬(self-loop 포함), 다른 모든 옵션, 사용 GPU
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    # 현재 최고 성능 Precision , early stopping 카운트
    cur_best_pre_0, stopping_step = 0, 0
    # 아담 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 손실, 정밀도(precision), 재현률(recall), ndcg, hit률
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        # 한 에폭당 필요한 배치 반복 횟수 계산
        n_batch = data_generator.n_train // args.batch_size + 1

        # 배치 단위로 학습 진행
        for idx in range(n_batch):
            # 학습 데이터 샘플링: 사용자, 긍정 아이템(선호함), 부정 아이템(선호하지 않음)
            # BPR(Bayesian Personalized Ranking) 학습을 위함
            users, pos_items, neg_items = data_generator.sample()
            
            # 모델의 Forward Pass 수행
            # u_g_embeddings: 사용자의 GNN 임베딩
            # pos_i_g_embeddings: 긍정 아이템의 GNN 임베딩
            # neg_i_g_embeddings: 부정 아이템의 GNN 임베딩
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            # BPR Loss(랭킹 손실)와 Regularization(정규화) 손실 계산
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            
            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 손실값 누적
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        # 10 에폭마다 성능 평가를 수행하지 않고 넘어감 (즉, 10의 배수 에폭마다 아래 코드 실행)
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        # 테스트 수행 (10 에폭마다)
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        # test 함수를 통해 Recall, Precision, NDCG 등의 지표 계산
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        # 로그 기록 저장
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        # Early Stopping 체크
        # Recall@20 (ret['recall'][0])이 5번 연속으로 증가하지 않으면 학습 중단
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping 조건 충족 시 학습 종료
        if should_stop == True:
            break

        # *********************************************************
        # 현재 성능이 최고일 때 모델의 가중치 저장
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)