# Code from: https://github.com/CausalML-Lab/PCMCI-Omega

import random 
import time
import numpy as np
from numpy import sum as sum
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from numpy import nan
from math import isnan
from copy import deepcopy
import math

def est_summary_casaul_graph(ar,icml_of_true_and_est,omega,N,tau_max_pcmci):
  new_ar= np.zeros(shape=(N,icml_of_true_and_est,N,tau_max_pcmci+1)) #if icml_of_ture_and_est=2, omega=2 new_ar=N,2,N,tau+1
  for i in range(N):
    omega_single=omega[i] #omega_single=2
    new_ar[i][0:omega_single]=ar[i][0:omega_single] #0:2=0:2
    replicate_num=int(icml_of_true_and_est/omega_single) #=1
    if replicate_num!=1:
      for j in range(replicate_num-1): #j=0
        new_ar[i][omega_single+j*omega_single:omega_single+(j+1)*omega_single]=ar[i][0:omega_single] #2:4=0:2
      #int(icml_of_true_and_est/omega_hat_single)
  return new_ar

def LCMofArray(a):
  lcm = a[0]
  for i in range(1,len(a)):
    lcm = lcm*a[i]//math.gcd(lcm, a[i])
  return lcm

def group_in_threes(slicable):
    for i in range(len(slicable)-2):
        yield slicable[i:i+3]

def turning_points(L):
    iloc=[]
    if L[0]<L[1]:
      iloc=[0]
    if len(np.where(L==next(x for x in L if not isnan(x)))[0])!=0:
      iloc.append(np.where(L==next(x for x in L if not isnan(x)))[0][0])
    for index, three in enumerate(group_in_threes(L)):
        if (three[0] > three[1] <= three[2]):
            #yield index + 1
            iloc.append(index+1)
    return iloc

def algorithm_v2_mci_(data,T,N,tau_max_pcmci,search_omega_bound, Omega):
  st = time.time()
  mask_all=np.empty(shape=(search_omega_bound,search_omega_bound,T,N))
  mask_all=mask_all+3 # The "+3" is a precaution in case there's an issue with generation, which would otherwise cause an error.

  for i in range(1, search_omega_bound+1):
    for j in range(0,i):
      a = [ [1]*N for i in range(T)]
      a =np.array(a,dtype=float)
      U = range(T)
      for t in range(N,T):
        if U[t]%i==j:
          a[t,:]=0
      mask_all[i-1][j]=a

  results_variable=np.zeros(shape=(N,search_omega_bound,search_omega_bound))
  results_variable[:][:][:] = nan
  num_edge1=np.zeros(shape=(N,search_omega_bound))
  num_edge1[:][:] = nan
  omega_hat1=np.zeros(shape=(N))
  num_edge2=np.zeros(shape=(N,search_omega_bound))
  num_edge2[:][:] = nan
  omega_hat2=np.zeros(shape=(N))
  num_edge3=np.zeros(shape=(N,search_omega_bound))
  num_edge3[:][:] = nan
  omega_hat3=np.zeros(shape=(N))
  num_edge4=np.zeros(shape=(N,search_omega_bound))
  num_edge4[:][:] = nan
  omega_hat4=np.zeros(shape=(N))
  turning_points_single_var=np.zeros(shape=(N,search_omega_bound))
  var_names=[]
  for i in range(N):
    var_names.append("$X^{}$".format(i))
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))}, 
                            var_names=var_names)
  parcorr = ParCorr(significance='analytic')
  pcmci = PCMCI(
      dataframe=dataframe, 
      cond_ind_test=parcorr,
      verbosity=0)
  pcmci.verbosity = 0
  results = pcmci.run_pcmci(tau_min=1,tau_max=tau_max_pcmci, pc_alpha=None, alpha_level=0.01)

  superset_bool=np.array(results["p_matrix"]<0.01, dtype=int)
  superset_dict_est = {}
  parents = {}
  for i in range(N):
      superset_dict_est[i] = dict()
      parents[i] = []
  for k in range(N):
    for i in range(N):
      for j in range(len(results["p_matrix"][0,0,:])):
        if superset_bool[i,k,j]==1:
          #  key='{}'.format(k)
          superset_dict_est[k].update({(i,-j): '-->'})
          parents[k].append((i,-j))

  #Superset_dict_est will be used in MCI tests and then select the Omega_hat.
  for i in range(1,search_omega_bound+1):#search omega=1,2
    # print("The current omega is {},total #Omega is {}".format(i,search_omega_bound))
    for j in range(0,i):
      # print("The current sub_sample is {},total #sub_sample under omega= {} is {}".format(j,i,i))
      dataframe = pp.DataFrame(data_run, mask= mask_all[i-1][j][N:T,:],
                              datatime = {0:np.arange(len(data_run))}, 
                              var_names=var_names)
      parcorr = ParCorr(significance='analytic',mask_type='y')
      pcmci = PCMCI(
          dataframe=dataframe, 
          cond_ind_test=parcorr,
          verbosity=0)
      pcmci.verbosity = 0
      results = pcmci.run_mci(link_assumptions=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci, 
                              parents=parents , 
                              alpha_level=0.01)
      for k in range(N):
        results_variable[k][i-1][j]=sum(sum(results["p_matrix"][:][k]<0.01))
        #print(results_variable)
  fix = deepcopy(results_variable)

  for i in range(1,search_omega_bound+1):#search omega=1,2
    for k in range(N):
      #num_edge[k][i-1]=np.sum(results_variable[k][i-1]) #i:omega
      num_edge1[k][i-1]=np.nanmean(results_variable[k][i-1])
      num_edge2[k][i-1]=np.nanmax(results_variable[k][i-1])
      num_edge3[k][i-1]=np.nanmin(results_variable[k][i-1])

  for k in range(N):
    # print(k)
    omega_hat1[k]=np.where(num_edge1[k,:]==num_edge1[k,:].min())[0][0]+1
    omega_hat2[k]=np.where(num_edge2[k,:]==num_edge2[k,:].min())[0][0]+1
    omega_hat3[k]=np.where(num_edge3[k,:]==num_edge3[k,:].min())[0][0]+1
    temp_list=[]
    for j in range(0,search_omega_bound-1):
      temp_list.append(turning_points(results_variable[k][:,j]))
    # print(temp_list)
    if len(list(set.intersection(*map(set, temp_list))))==0:
      omega_hat4[k]=nan
    else:
      turning_points_single_var[k]=list(set.intersection(*map(set, temp_list)))[0]
      omega_hat4[k]=np.amin(turning_points_single_var[k])+1
  for k in range(N):
    for i in range(1,search_omega_bound+1):#search omega=1,2
     for j in range(0,i):
        if results_variable[k][0][0]<=np.min((results_variable[k][1][0],results_variable[k][1][1])):
         omega_hat4[k]=1
  for k in range(N):
    if np.isnan(omega_hat4[k]):
      omega_hat4[k]=omega_hat2[k]
  #suppose omega_hat4=[2,2,1,1,1]
  # print("Omega hat is {}".format(omega_hat4))
  # print("Omega is {}".format(Omega))

  omega_hat4=np.array(omega_hat4,dtype=int)
  merge_omega=np.concatenate((omega_hat4,Omega))
  tem_array=np.zeros(shape=(N,LCMofArray(merge_omega),N,tau_max_pcmci+1))
  # union_matrix=np.zeros(shape=(N,N,tau_max_pcmci+1))
  omega_hat4=np.array(omega_hat4,dtype=int)
  for k in range(N): #for one specifc variable
    # print("The current target variable is {},total #Variable is {}".format(k,N))
    # for i in range(omega_hat4[k]): #i represent the omega_hat for this variable;Omega=2 then i = 1.
    # print("The current target variable is {},the Omega_hat is {}".format(k,Omega[k]))
    for j in range(0,omega_hat4[k]): #different parents set; if Omega=2,then i=1, then j=0,1;
        # print("The current target variable is {},the set index is {}".format(k,j))
      dataframe = pp.DataFrame(data_run, mask= mask_all[omega_hat4[k]-1][j][N:T,:],
                                datatime = {0:np.arange(len(data_run))}, 
                                var_names=var_names)
      parcorr = ParCorr(significance='analytic',mask_type='y')
      pcmci = PCMCI(
          dataframe=dataframe, 
          cond_ind_test=parcorr,
          verbosity=0)
      pcmci.verbosity = 0
      #superset_dict_est
      results = pcmci.run_mci(link_assumptions=superset_dict_est,tau_min=1,tau_max=tau_max_pcmci,parents=parents, alpha_level=0.01)
      #results = pcmci.run_mci(tau_min=1,tau_max=tau_max_pcmci,parents=superset_dict_est, alpha_level=0.01)
      #p matrix = [N*N*(tau_max+1)]
      tem_array[k][j]=results['p_matrix'][:,k,:]<0.01
      tem_array[k][j]=np.asmatrix(tem_array[k][j])

  et = time.time()
  # get the execution time
  elapsed_time = et - st
  return tem_array, omega_hat4, superset_bool, elapsed_time


def pcmci_omega(ts: np.ndarray, tau_max: int) -> np.ndarray:
    search_omega_bound=10
    # search_tau_bound=20
    tau_max_pcmci=tau_max # assuming we know the true tau_max
    N, T = ts.shape
    Omega_bound=1

    Omega = [0]*N
    for i in range(N):
        Omega[i]=random.randint(1,Omega_bound)
    if max(Omega)<Omega_bound:
        selected_one = random.randint(1,N)
        Omega[i]=Omega_bound

    algorithm_v2_mci_results = algorithm_v2_mci_(ts.T, T, N, tau_max_pcmci,search_omega_bound, Omega)
    tem_array=algorithm_v2_mci_results[0]
    omega_hat4=algorithm_v2_mci_results[1]
    
    merge_omega=np.concatenate((omega_hat4,Omega))
    lcm=LCMofArray(merge_omega)
    tem_array1=deepcopy(tem_array)
    est_summary_matrix=est_summary_casaul_graph(tem_array1,lcm,omega_hat4,N, tau_max)
    est_summary_matrix = est_summary_matrix.squeeze().max(-1)
    np.fill_diagonal(est_summary_matrix, 0)
    return est_summary_matrix
