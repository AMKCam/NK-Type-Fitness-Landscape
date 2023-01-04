#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:35:15 2018

@author: angela-kirykowicz
"""
#Model implements probabilistic removal and reproduction of members of pop
#Fitness assignment per position is now random - more realistic
# i.e position 0 fit assignment is (A=1,G=0,C=1,T=0), position 1 fit assignment is (A=0,G=1,C=1,T=0), etc
#Only 1 member of pop reproduces at a time
#Pop size kept to certain number e.g 100 members, 1000 members - set in running of model
#Removal of pop membersis random - 1 member randomly removed from pop
#Reproduction of member in pop based on selecting a member from normal distn centered on highest fitness
#N random mutations implemented
#Add epistatsis into model - K = 2

import random
import json
import numpy as np
import matplotlib.pyplot as plt
import math as m
import statistics as s
import itertools as itool
#from pylab import figure, axes, pie, title, show

def ran_dna_seq(r,n):
    """
    Function generates n random dna sequences of length r.
    Choose length as multiple of 3.
    """
    s = ''
    count = 0
    d = {0:'A', 1:'T', 2:'C', 3:'G'}
    dna_seq_list = []
    # Generates random DNA sequences and stores them in a list
    while count < n: #Set number of random sequences
        s = '' #start codon
        for letter in range(0,r): #Set length of sequence
            num = random.randint(0,3)
            s += str(d[num])
        dna_seq_list.append(s)
        count += 1
    return dna_seq_list


def single_epi_fit_dic(r):
    """
    Function makes fitness dictionary.
    Input - desired length of sequence (r)
    Output - Random assignment of 0 or 1 to each nucleotide at each position 0 - r
    Start codon positions - already assigned fixed fitnesses
    """
    nuc_list = ['A', 'G', 'T', 'C']
    fit_dic = {}  
    for i in range(0,r):
        nuc_dic = {}
        for letter in nuc_list:
            num = random.randint(0,1)
            nuc_dic[letter] = num
        fit_dic[i] = nuc_dic
    return fit_dic

            
def pair_epi_fit_dic(r):
    """
    Creates fitness dictionary for all permutations of nucleotides
    Each pair is randomly assigned 0 or 1
    """
    pair_fit_dic = {}
    pair_list = []
    for elem in itool.permutations('ATCG',2):
        pair_list.append(elem)
    pair_list.append(('A','A'))
    pair_list.append(('C','C'))
    pair_list.append(('G','G'))
    pair_list.append(('T','T'))
    for i in range(0,r):
        pair_dic = {}
        for pair in pair_list:
            num = random.randint(0,1)
            pair_dic[pair] = num
        pair_fit_dic[i] = pair_dic
    return pair_fit_dic

def pair_unique_sol(epi_dic, sing_dic,r):
    count = 0
    count_list = []
    val_list = []
    sol_sing_dic = {}
    corr_pair_dic = {}
    d = {}
    mult = 1
    for key, value in sing_dic.items():
        val_list = []
        count_0 = sum(item == 0 for item in value.values())
        if count_0 < 4:
            for k, v in value.items():
                if v == 1:
                    val_list.append(k)
        elif count_0 == 4:
            for k, v in value.items():
                val_list.append(k)
        sol_sing_dic[key] = val_list
    #print(sol_sing_dic)
    for i in range(0,r):
        d = {}
        if i < r-1:
            for j in itool.product(sol_sing_dic[i], sol_sing_dic[i+1]):
                d[(j[0], j[1])] = 1
            corr_pair_dic[i] = d
        if i == r-1:
            for j in itool.product(sol_sing_dic[i], sol_sing_dic[0]):
                d[j[0], j[1]] = 1
            corr_pair_dic[i] = d
    #print(corr_pair_dic)
    #print(epi_dic)
    for key, value in epi_dic.items():
        for k, v in corr_pair_dic.items():
            count = 0
            if k == key:
                for k2 in value.keys():
                    if k2 in v.keys():
                        if value[k2] == 1:
                            #print(key,k2,value[k2])
                            count += 1
                count_list.append(count)
    for elem in count_list:
        if elem > 0: #next best solution
            mult *= elem
    return mult
                    
def pair_unique_sol_count(runs,r):
    p_list = []
    i_list = []               
    for i in range(runs):
        i_list.append(i)
        pair = pair_epi_fit_dic(r)
        sing = single_epi_fit_dic(r)
        p = pair_unique_sol(pair,sing,r)
        p_list.append(p)
    plt.plot(i_list,p_list)
    mean = s.mean(p_list)
    print(mean)
    plt.savefig("Unique_Solutions_next_best_K_1_len_60_1000_runs.png",dpi=300)

        


def three_epi_fit_dic(r):
    """
    Creates fitness dictionary for all permutations of nucleotides (64 possible)
    Each tuple of 3 nucleotides is randomnly assigned 0 or 1 for each position of sequence with length r
    """
    three_fit_dic = {}
    three_list = []
    for elem in itool.permutations('ATCG',3):
        three_list.append(elem)
    three_list_A = [('A','A','A'),('A','A','T'),('A','A','C'),('A','A','G'),('A','T','T'),('A','G','G'),('A','C','C'),('A','T','A'),('A','G','A'),('A','C','A')]
    three_list_T = [('T','T','T'),('T','T','A'),('T','T','C'),('T','T','G'),('T','A','A'),('T','C','C'),('T','G','G'),('T','A','T'),('T','G','T'),('T','C','T')]
    three_list_C = [('C','C','C'),('C','C','A'),('C','C','G'),('C','C','T'),('C','A','A'),('C','T','T'),('C','G','G'),('C','A','C'),('C','T','C'),('C','G','C')]
    three_list_G = [('G','G','G'),('G','G','A'),('G','G','C'),('G','G','T'),('G','T','T'),('G','A','A'),('G','C','C'),('G','A','G'),('G','T','G'),('G','C','G')]
    three_list.extend(three_list_A)
    three_list.extend(three_list_T)
    three_list.extend(three_list_C)
    three_list.extend(three_list_G)
    for i in range(0,r):
        three_dic = {}
        for three in three_list:
            num = random.randint(0,1)
            three_dic[three] = num
        three_fit_dic[i] = three_dic
    return three_fit_dic
            

def three_epi_fit_assign(nuc,three_fit,r):
    """
    Function assigns fitness values for 2 nearesr neighbours
    e.g position 1, consider 123, 234, 345 combinations
    Inputs: three_fit dictionary, nucleotide sequences (nuc) and length of sequence (r)
    Output: dictionary with fitness assignments (value between 0 and 1)
    """
    three_dic = {}
    for seq in nuc:
        val_1 = 0
        val_2 = 0
        val_3 = 0
        seq_list = list(seq)
        for i in range(0,r):
            if i == r-1:
                for key, value  in three_fit.items():
                    if i == key:
                        val_1 += value[(seq_list[i],seq_list[0],seq_list[1])]
                       
            elif i == r-2:
                for key, value  in three_fit.items():
                    if i == key:
                        val_2 += value[(seq_list[i],seq_list[i+1],seq_list[0])]
                        
            else:
                for key, value  in three_fit.items():
                    if i == key:
                        val_3 += value[(seq_list[i],seq_list[i+1],seq_list[i+2])]
        val = val_1 + val_2 + val_3
        val_prob = round(val/r,4)
        three_dic[seq] = val_prob
    return three_dic
                


def epi_fit_assign(nuc,epi_fit,r):
    """
    Function assigns fitness values based on nearest neighbours
    e.g position 1, consider 10, 01, 12, and 21 combinations
    Inputs: epi_fit dictionary, nuceleotide sequences (nuc) and length of sequence (r)
    Output: dictionary with fitness assignments (value between 0 and 1)
    """
    epi_dic = {}
    for seq in nuc:
        val_1 = 0
        val_2 = 0
        seq_list = list(seq)
        for i in range(0,r):
            if i == r-1:
                for key, value in epi_fit.items():
                    if i == key:
                        val_1 += value[(seq_list[i],seq_list[0])]
            else:
                for key, value in epi_fit.items():
                    if i == key:
                        val_2 += value[(seq_list[i],seq_list[i+1])]
        val = val_1 + val_2
        val_prob = round(val/r,4)
        epi_dic[seq] = val_prob
    return epi_dic

def single_fit_assign(func,seq_fit):
    """
    Function makes fitness dictionary.
    Input - desired length of sequence (r)
    Output - Random assignment of 0 or 1 to each nucleotide at each position 0 - r
    Start codon positions - already assigned fixed fitnesses
    """
    fit_assign_dic = {}
    for seq in func: #loops over seq list generated by ran_dna_seq function
        fit_sum = 0
        fit_prob = 0
        for i, elem in enumerate(seq): #loops over each seq element
            for key,value in seq_fit.items():
                if i == key:
                    fit_sum += value[elem] #sums each fitness assignment over seq
        fit_prob = round((1/len(seq)) * fit_sum,4) #calculates prob for seq
        #fit_per = round(fit_prob * 100,3)
        fit_assign_dic[seq] = fit_prob #assigns seq prob to dictionary
    return fit_assign_dic

def total_fit_assign(epi_assign,single_assign,three_assign):
    """
    Calculates total fitness based on average fitness contributions from pair interactions
    and single position
    """
    new_dic = {}
    for key,value in three_assign.items():
        if (key in single_assign.keys()) and (key in epi_assign.keys()):
            new_value = round((three_assign[key] + single_assign[key] + epi_assign[key])/3,3)
        new_dic[key] = new_value
    return new_dic


#Initial Fitness Assignment
#ran_seq = ran_dna_seq(21,10)
#ran_seq_fit = fit_assign(ran_seq,fit_dic) #set initial fitness of population

#n = 100 #number of initial sequences
#Kill member with lowest fitness
def ran_remove_member(assign,pop_size):
    """
    Function removes a sequence randomly.
    Each sequence assigned a number.
    A random number is drawn - if the member number matches this number, it is removed from pop.
    """
    pop_dic = {}
    red_dic = {}
    new_assign_dic = {}
    pop_num = 0
    size_pop = len(assign.values())
    if size_pop >= pop_size:
        for key in assign.keys(): #each member assigned a number from 0 to #members
            pop_dic[key] = pop_num
            pop_num += 1
        num_list = random.sample(range(0,size_pop),1) #draw random number
        for key, value in pop_dic.items():
            if value not in num_list:
                red_dic[key] = value #don't include one member
        for key, value in assign.items():
            if key in red_dic.keys():
                new_assign_dic[key] = value #get back fitness values for the keys not removed
    else:
        new_assign_dic.update(assign)
    return new_assign_dic

#Decide reproduction seq
def reproduction(assign):
    """
    Function assigns sequence to reproduce using probabilistic method.
    Sequence fitnesses normalised.
    Random number drawn from normal distribution centered on highest fitness in pop (variance 1)
    If member normal fitness is greater than random num, it is added to highest_fit_dic
    highest_fit_dic can only take one member.
    Drawing of random number repeated until a member from pop fills highest_fit_dic
    """
    num_mem = len(assign.values())
    rep_assign_copy = assign.copy()
    #print(num_mem)
    highest_fit_dic = {}
    high_fit_norm = {}
    max_fit = max(assign.values()) #maximum fitness is highest fitness value
    value_list = []
    for value in assign.values():
        value_list.append(value)
    mean = s.mean(value_list)
    var = s.variance(value_list)
    rep_norm_dic = {}
    if var > 0:
        max_fit_norm = round((max_fit - mean)/m.sqrt(var/num_mem))
        for key,value in assign.items():
            rep_norm_val = round((value - mean)/m.sqrt(var/num_mem),3)
            rep_norm_dic[key] = rep_norm_val
   # print(rep_norm_dic)
        while len(high_fit_norm.values()) < 1:
            rep_st_norm = round(random.gauss(max_fit_norm,1),3)
            for key,value in rep_norm_dic.items():
                if len(high_fit_norm.values()) < 1:
            #print(rep_st_norm)
                    if value > rep_st_norm:
                        high_fit_norm[key] = value
    #print(high_fit_norm)
        for key,value in rep_assign_copy.items():
            if key in high_fit_norm:
                highest_fit_dic[key] = value
    else:
        while len(highest_fit_dic.values()) < 1:
            for key, value in rep_assign_copy.items():
                if len(highest_fit_dic.values()) < 1:
                    highest_fit_dic[key] = value
                
    return highest_fit_dic

#Mutate seq with highest fit
dna_dic = {0:'A', 1:'T', 2:'C', 3:'G'} #dictionary of nucleotides
#mut_seq = reproduction_first_seq(assign) #sequences to mutate (highest fitness)
def mutate_seq(mut_seq,mut_num):
    """
    Function mutates members from input function. 
    Two random mutations per sequence randomly along the sequence.
    """ 
    mutate_seq_list = [] 
    count = 0
    for key in mut_seq.keys():
        #print(key)
        list_key = list(key)
        #print(list_key)
        while count < mut_num:
            num = random.randint(0,len(list_key))
            #print(num)
            mut_num = random.randint(0,3)
            #print(mut_num)
            for i,j in enumerate(list_key):
                if i == num:
                    j = dna_dic[mut_num]
                    list_key[i] = j
                    #print(list_key)
                    string_key = "".join(list_key)
                    #print(string_key)
                    count += 1
        mutate_seq_list.append(string_key)
        #print(mutate_seq_list)
    return mutate_seq_list


#Hamming distance - can only calculate for limited pop
def hamming(seq,len_seq):
    """
    Function calculates average Hamming distance in pop
    """
    num_seq = len(seq.values()) #calculate number of sequences in pop
    #print(num_seq)
    seq_list = []
    arr = np.empty((num_seq,len_seq)) #create an empty array to add in sequences
    val_dic = {'A':1, 'C':2, 'G':3, 'T':4} #create dictionary to convert strings in seq to nums
    l = []
    d = {}
    H = {}
    dist = 0
    H_distance = 0
    for key in seq.keys():
        seq_list.append(key) #take out sequences from dictionary
    for seq in seq_list:
        list_seq = list(seq) #convert sequences to list
        l = []
        for elem in list_seq: #loop over each element in seq
            val = val_dic[elem] #replace seq element with number using val_dic
            l.append(val) #append number to list
        d[seq] = l #add numbers to another dic, using old sequence as a key
    #print(d)
    i = 0
    for value in d.values():
        arr[i,:] = value #append value to empty array
        i += 1 #next row
    #print(arr)
    for i in range(0,int(len_seq)):
        num_1 = str(arr[:,i]).count('1') #count total 1's going along the matrix columns
        num_2 = str(arr[:,i]).count('2') #count total 2's going along matrix columns
        num_3 = str(arr[:,i]).count('3') #count total 3's going along matrix columns
        num_4 = str(arr[:,i]).count('4') #count total 4's going along matrix columns
        if (num_1 == num_seq) or (num_2 == num_seq) or (num_3 == num_seq) or (num_4 == num_seq):
            dist = 0 #if column has same num, assign dist of 0
            H[i] = dist #append value to H dic
        else:
            dist = 1 #else column has diff nums, assign dist of 1
            H[i] = dist #append value to H dic
    #print(H)  
    for value in H.values():
        H_distance += value #add values 1 or 0 assigned to columns - Hamming distance
    return H_distance

        
def consensus_seq(seq,len_seq):
    num_seq = len(seq.values()) #calculate number of sequences in pop
    print(num_seq)
    seq_list = []
    consen = {}
    arr = np.empty((num_seq,len_seq)) #create an empty array to add in sequences
    val_dic = {'A':1, 'C':2, 'G':3, 'T':4} #create dictionary to convert strings in seq to nums
    l = {}
    d = {}
    for key in seq.keys():
        seq_list.append(key) #take out sequences from dictionary
    for seq in seq_list:
        list_seq = list(seq) #convert sequences to list
        l = []
        for elem in list_seq: #loop over each element in seq
            val = val_dic[elem] #replace seq element with number using val_dic
            l.append(val) #append number to list
        d[seq] = l #add numbers to another dic, using old sequence as a key
    #print(d)
    i = 0
    for value in d.values():
        arr[i,:] = value #append value to empty array
        i += 1 #next row
    for i in range(0,int(len_seq)):
        num_1 = str(arr[:,i]).count('1') #count total 1's going along the matrix columns
        num_2 = str(arr[:,i]).count('2') #count total 2's going along matrix columns
        num_3 = str(arr[:,i]).count('3') #count total 3's going along matrix columns
        num_4 = str(arr[:,i]).count('4') #count total 4's going along matrix columns
        if num_1 == num_seq:
            consen[i] = ['A']
        elif num_2 == num_seq:
            consen[i] = ['C']
        elif num_3 == num_seq:
            consen[i] = ['G']
        elif num_4 == num_seq:
            consen[i] = ['T']
        elif (num_1 == 0) and (num_2 != 0) and (num_3 != 0) and (num_4 != 0):
            consen[i] = ['C', 'G', 'T']
        elif (num_1 == 0) and (num_2 == 0) and (num_3 != 0) and (num_4 != 0):
            consen[i] = ['G', 'T']
        elif (num_1 == 0) and (num_2 != 0) and (num_3 == 0) and (num_4 != 0):
            consen[i] = ['C', 'T']
        elif (num_1 == 0) and (num_2 != 0) and (num_3 != 0) and (num_4 == 0):
            consen[i] = ['C', 'G']
        elif (num_1 != 0) and (num_2 != 0) and (num_3 != 0) and (num_4 == 0):
            consen[i] = ['A', 'C', 'G']
        elif (num_1 != 0) and (num_2 != 0) and (num_3 == 0) and (num_4 == 0):
            consen[i] = ['A', 'C']
        elif (num_1 != 0) and (num_2 == 0) and (num_3 != 0) and (num_4 == 0):
            consen[i] = ['A', 'G']
        elif (num_1 != 0) and (num_2 == 0) and (num_3 == 0) and (num_4 != 0): 
            consen[i] = ['A', 'T']
        elif (num_1 != 0) and (num_2 == 0) and (num_3 != 0) and (num_4 != 0):
            consen[i] = ['A', 'G', 'T']
        elif (num_1 != 0) and (num_2 != 0) and (num_3 == 0) and (num_4 != 0):
            consen[i] = ['A', 'C', 'T']
        elif (num_1 != 0) and (num_2 != 0) and (num_3 != 0) and (num_4 != 0):
            consen[i] = ['A', 'C', 'G', 'T']
    return consen

def dna_fit_model(r,n,sim,pop_size,gen):
    """
    Function implements dna fitness model.
    Set length of sequence (r), pop size (n), number of simulations to run (sim), and 
    number of generations (gen)
    """
    num_sim = 0
    con_sim = {}
    pair_fit = pair_epi_fit_dic(r)
    sing_fit = single_epi_fit_dic(r)
    three_fit = three_epi_fit_dic(r)
    while num_sim < sim: #run simulation n times
        #print("simulation ", num_sim)
        gen_count = 0 #initial random sequences
        gen_dic_seq = {}
        gen_dic_fit = {}
        pop_seq = []
        pop_fit = []
        #h_dist_list = []
        init_seq = ran_dna_seq(r,n) #up to 100 seq per gen
        print("initial sequences are ",init_seq)
        init_fit_sing = single_fit_assign(init_seq,sing_fit)
        init_fit_pair = epi_fit_assign(init_seq,pair_fit,r)
        init_fit_three = three_epi_fit_assign(init_seq,three_fit,r)
        init_fit_tot = total_fit_assign(init_fit_pair,init_fit_sing,init_fit_three)
        for key, value in init_fit_tot.items():
            pop_seq.append(key)
            pop_fit.append(value) 
        gen_dic_seq[gen_count] = pop_seq
        gen_dic_fit[gen_count] = pop_fit
        print("initial fitnesses are ",init_fit_tot)  
        #init_h_dist = hamming(init_fit_tot,r)
        #h_dist_list.append(init_h_dist)
        #print("Initial pop hamming distance is ", init_h_dist)
        ran_rem = ran_remove_member(init_fit_tot,pop_size)
        print("initial pop is ", ran_rem)
        gen_count += 1
        while gen_count < gen:   #initially run it for 10 generations
            pop_seq = []
            pop_fit = []
            rep = reproduction(ran_rem)
            print("generation ", gen_count, " sequences to reproduce are ",rep)
            mut = mutate_seq(rep)
            print("generation ", gen_count, " new mutated sequences are ", mut)
            new_fit_sing = single_fit_assign(mut,sing_fit)
            new_fit_pair = epi_fit_assign(mut,pair_fit,r)
            new_fit_three = three_epi_fit_assign(mut,three_fit,r)
            new_fit_tot = total_fit_assign(new_fit_pair,new_fit_sing,new_fit_three)
            print("generation ", gen_count, " new sequence fitnesses are ", new_fit_tot)
            ran_rem.update(new_fit_tot)
            #h_dist = hamming(ran_rem,r)
            #h_dist_list.append(h_dist)
            #print("Generation ", gen_count, " hamming distance is ", h_dist)
            ran_rem = ran_remove_member(ran_rem,pop_size)
            print("generation ", gen_count, " new population is ", ran_rem)
            for key, value in ran_rem.items():
                pop_seq.append(key)
                pop_fit.append(value) 
            gen_dic_seq[gen_count] = pop_seq
            gen_dic_fit[gen_count] = pop_fit
            gen_count += 1
  #  seq_len.append(gen_dic_seq)
   # seq_len.append(gen_dic_fit)
   # seq_len_dic[n] = seq_len
   # n += 1
        #Graph for average fitness
        x = []
        con = consensus_seq(ran_rem,r)
        title = "Generation " + str(num_sim)
        con_sim[title] = con
        for key in gen_dic_fit.keys():
            x.append(key)
        y = []
        i = 0
        for key,value in gen_dic_fit.items():
            if key == i:
                mean = round(s.mean(value),3)
            y.append(mean)
            i += 1
            
        #with open('Fit_model_5_10_mut_epi_K_2_5_sims_consensus_seq_len_300_100_seq_per_gen_10000_gen.json', 'w') as f:
            #json.dump(con_sim, f)
   # sim_round[num_sim] = gen_dic_fit
        #plt.plot(x,y) ; plt.xlabel("Generation"); plt.ylabel("Average Fitness")
        #plt.plot(x,h_dist_list) ; plt.xlabel("Generation") ; plt.ylabel("Average Hamming Distance")
        num_sim +=1

    #plt.savefig("Fit_model_5_10_mut_epi_K_2_5_sims_len_300_100_seq_per_gen_10000_gen.png", dpi=300)
    #plt.savefig("Fit_model_5_10_mut_epi_K_2_5_sims_Hamming_distance_len_300_10_seq_per_gen_1000_gen.png", dpi=300)
    #plt.savefig("Fit_model_2_1_sim_Fitness_Hamming_Dist_len_99_100_seq_init_10_seq_per_gen_1000_gen_T_and_C_0_fit.png", dpi=300)
