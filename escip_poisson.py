# ESCIP-Areal
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import multiprocessing as mp
import math
from tqdm import tqdm
from collections import Counter
from functools import cmp_to_key
import random
cpus = mp.cpu_count() - 1

class Model:
    def __init__(self, cases: pd.DataFrame, background: pd.DataFrame, epsilon:float, x = 'x', y = 'y', cases_name="cases", merge_key = "key",background_name="background", t=None, dist_metric = ''):
        if background is None:
            self.df = cases[[x, y, cases_name, background_name]]
        else:
            self.df = pd.mergr(cases, background, merge_key)[[x, y, cases_name, background_name]]
        self.coords = self.df[[x, y]].values
        len_data = len(cases)
        self.l1 = len_data  
        self.epsilon = epsilon
        self.sum_background = self.df[background_name].sum()
        self.sum_cases = self.df[cases_name].sum()
        self._c_b_ratio = self.sum_cases/self.sum_background
        self._c_b_ratio_exp = math.exp(-self._c_b_ratio)
        self.df.columns = ['x', 'y', 'cases', 'background']
        if t is None:    # spatial mode (not spatio-temporal mode)
            # print("Start Indexing...")
            self.index = KDTree(np.row_stack((self.coords)))
            # print("Index Created!")
        else:
            # to be implemented
            print('Not Implemented')
            exit()
        
    def radius_query(self, r:float, x: np.ndarray):
        return self.index.query_radius(x, r, count_only = False, return_distance = False, sort_results = False)

    def _calc_core_points_parallel_(self, i):
        background = self.df['background'].iloc[i]
        cases = self.df['cases'].iloc[i]
        lambda0 = background * self._c_b_ratio
        # lambda0_j = 1
        p = 0
        # j_fac = 1
        for j in range(cases):
            # if j != 0:
            #     j_fac *= j
            #     lambda0_j *= lambda0
            p += math.pow(lambda0, j) * self._c_b_ratio_exp / math.factorial(j)
        
        if self.tail == 'two':
            flag = abs(p-0.5)>=(0.5-self.alpha/2)
        elif self.tail == 'left':
            flag = p <= self.alpha
        else:
            flag = p >= 1 - self.alpha
        if flag:
            self.cor_pts.append(i)
            pts_within_e = self.radius_query(self.epsilon, [self.coords[i,:]])[0]
            self.adj_pts_dict[i] = pts_within_e
            

    def calc_core_points(self, epsilon: float):
        manager = mp.Manager()
        self.cor_pts = manager.list()
        self.adj_pts_dict = manager.dict()
        pool = mp.Pool(processes=cpus)
        pool.imap(self._calc_core_points_parallel_, [i for i in range(self.l1)])
        pool.close()
        pool.join()


    def register_model(self, epsilon = None, alpha = 0.05, n_mc = 100, tail="two"):
        if tail not in ['two', 'left', 'right', 't', 'l', 'r']:
            print('The model must use one of one- or two-tailed tests.')
        if tail in ['two','t']:
            self.tail = 'two'
        elif tail in ['left', 'l']:
            self.tail = 'left'
        elif tail in ['right', 'r']:
            self.tail = 'right'
        if epsilon == None:
            epsilon = self.epsilon
        else:
            self.epsilon = epsilon
        self.alpha = alpha
        self.calc_core_points(epsilon)
    

    def expansion(self):
        clus_id = [-1] * (self.l1)
        clus_count = 0
        clusters = [[]]
        for i in self.cor_pts:
            clus_id[i] = 0
        for i in self.cor_pts:
            if clus_id[i] == 0:
                # check if current point belongs to any group
                flag = 0
                for j in self.adj_pts_dict[i]:
                    if clus_id[j] <= 0:
                        continue
                    else:
                        clus_id[i] = clus_id[j]
                        clusters[clus_id[j]].append(i)
                        flag = 1
                        break
                if flag == 0: # new cluster
                    clus_count += 1
                    clus_id[i] = clus_count
                    clusters.append([])
                    for j in self.adj_pts_dict[i]:
                        clus_id[j] = clus_count
                        clusters[clus_count].append(j)
        # self.clus_id = clus_id
        # self.clus_count = clus_count
        clusters = [i for i in clusters if len(i) >= 1]
        self.clusters_expansion = clusters


    def extract_clusters(self, minimum_len = 2, n_mcs = 1000):
        clusters = [i for i in self.clusters_expansion if len(i) >= minimum_len]
        n_clusters = len(clusters)
        clus_per_cpu = math.ceil(n_clusters/cpus)
        manager = mp.Manager()
        p = manager.Array('d', [-1] * n_clusters)
        lw = manager.Array('d', [-10e4] * n_clusters)
        p_list = []

        def test_clusters(i_mp, p, lw):
            upper = min((i_mp+1) * clus_per_cpu, n_clusters)
            for i in range(i_mp * clus_per_cpu, upper):
                log_likelihood = []
                sum_background_i = self.df['background'].iloc[clusters[i]].sum()
                lambda_i = sum_background_i * self._c_b_ratio
                sum_cases_i = self.df['cases'].iloc[clusters[i]].sum()
                if sum_cases_i <= 0:
                    sum_cases_i = 0.1
                if sum_background_i <= 0:
                    sum_background_i = 0.1
                if lambda_i == 0:
                    lambda_i += 0.1
                if sum_cases_i/lambda_i <= -1:
                    print(sum_cases_i, lambda_i)
                if (self.sum_cases - sum_cases_i)/(self.sum_cases - lambda_i) <= -1:
                    print("hsadghjebajksdab")
                l = sum_cases_i * math.log1p(sum_cases_i/lambda_i) + (self.sum_cases - sum_cases_i) * math.log1p((self.sum_cases - sum_cases_i)/(self.sum_cases - lambda_i))
                for _ in range(n_mcs-1):
                    sum_cases_i = np.random.poisson(lambda_i)
                    if sum_cases_i == 0:
                        sum_cases_i += 0.1
                    ll = sum_cases_i * math.log1p(sum_cases_i/lambda_i) + (self.sum_cases - sum_cases_i) * math.log1p((self.sum_cases - sum_cases_i)/(self.sum_cases - lambda_i))
                    log_likelihood.append(ll)
                p_ = sum(log_likelihood >= l)/n_mcs
                p[i] = p_
                lw[i] = l

        for i_mp in range(cpus):
            p_list.append(mp.Process(target=test_clusters, args=(i_mp, p, lw)))
            p_list[i_mp].start()
        for ppp in p_list:
            ppp.join()

        self.clus_info = []
        for i in range(n_clusters):
            self.clus_info.append({'id': i, 'points': clusters[i], 'p': p[i], 'lw': lw[i]})
        
    def dump_clusters(self, path:str, lim=40, info_csv=True, merge=False):
        if self.clus_info is None:
            print("Cluster information is not extracted. Use `Model.extract_clusters`")
            return
        elif len(self.clus_info) == 0:
            print("No cluster extracted.")
            return
        def comp_cluster_info(c1, c2):
            if c1['lw'] > c2['lw']:
                return -1
            elif c1['lw'] < c2['lw']:
                return 1
            if c1['p'] < c2['p']:
                return -1
            elif c1['p'] >= c2['p']:
                return 1


        clus_info = sorted(self.clus_info, key = cmp_to_key(comp_cluster_info))
        if len(clus_info) > lim:
            clus_info = clus_info[:lim]
        info = pd.DataFrame(clus_info)
        if info_csv == True:
            info.to_csv(path + '_info.csv')

        merged = []
        for i in range(len(clus_info)):
            pts_idx = clus_info[i]['points']
            # print(clus_info[i]['points']) ## 
            # print('cluster ' + str(i) + ' has ' + str(pts_idx) + ' points.') ##
            clus_pd = self.df.iloc[pts_idx].copy()
            clus_pd['escip_cluster_id'] = clus_info[i]['id']
            if merge == False:
                clus_pd.to_json(path + '{:02d}_{:04d}.json'.format(i, clus_info[i]['id']), index = False, orient='table')
            if i == 0:
                merged = clus_pd.copy()
            else:
                merged = merged.append(clus_pd)
        merged.to_json(path + '.json')

        return info

        
