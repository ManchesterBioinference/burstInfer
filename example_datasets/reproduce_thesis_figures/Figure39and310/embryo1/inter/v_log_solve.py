# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:52:32 2019

@author: Jon
"""
import numpy as np
import itertools
import scipy

def v_log_solve(m_lg, m_sgn, b_lg, b_sgn):
    
    # https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python    
    def sub2ind(array_shape, rows, cols):
        return rows*array_shape[1] + cols
    
    def permutation_parity(lst):
        I = scipy.sparse.eye(len(lst)).toarray()
        parity = np.linalg.det(I[:,lst-1])
        return parity
    
    def log_sum_exp(arr, signs):
        arr_max = np.max(arr[:,:])
        term2_array = np.multiply(signs, np.exp(arr-arr_max))
        term2 = np.sum(np.ravel(term2_array))
        logsum = np.array([arr_max + np.log(np.abs(term2)), np.sign(term2)])
        return logsum
    
    def find_perms(m_lg):
        n = np.size(m_lg, 0)
        perm_input = np.arange(1, n+1)
        perm_list = list(itertools.permutations(perm_input))
        perm_list2 = np.array(perm_list)
        perm_num = np.size(perm_list2, 0)
        return perm_num, perm_list2
    
    def log_determinant(m_lg, m_sgn):

        perm_answer = find_perms(m_lg)
        perm_num = perm_answer[0]
        perm_list2 = perm_answer[1]
        
        logs = np.zeros((perm_num, 1))
        signs = np.zeros((perm_num, 1))
        
        for i in np.arange(0, perm_num):
            rows = np.arange(0, perm_num)
            cols = perm_list2[i,:]

            
            ind1 = sub2ind(np.array([2,2]), rows[0], cols[0])
            ind2 = sub2ind(np.array([2,2]), rows[1], cols[1])
            
            
            raveled = np.ravel(m_sgn, order = 'C')
            raveled = np.expand_dims(raveled, axis = 1)
            signs[i,0] = permutation_parity(cols) * np.prod(np.concatenate((raveled[ind1-1,], raveled[ind2-1,]), axis = 0))
            raveled2 = np.ravel(m_lg, order = 'C')
            raveled2 = np.expand_dims(raveled2, axis = 1)
            logs[i,0] = np.sum(np.concatenate((raveled2[ind1-1,], raveled2[ind2-1,]), axis = 0), axis = 0)
        
        
        log_det = log_sum_exp(logs, signs)[0]
    
        det_m = log_det
        return det_m, np.sign(det_m)
    
    n = np.size(m_lg, 0)
    v_lgs = np.zeros((n, 1))
    v_sgns = np.zeros((n, 1))

    det_m = log_determinant(m_lg, m_sgn)

    for j in np.arange(0,n):
        m_log_j = m_lg.copy()        
        m_log_j[:,j] = b_lg
        
        
        m_sgn_j = m_sgn
        m_sgn_j[:,j] = b_sgn
        
        det_j = log_determinant(m_log_j, m_sgn_j)
        v_lgs[j,] = det_j[0] - det_m[0]
        v_sgns[j,] = det_j[1] * det_m[1]
    
    v_sln = np.concatenate((np.transpose(v_lgs), np.transpose(v_sgns)), axis = 0)
    return v_sln
