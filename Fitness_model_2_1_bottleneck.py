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
#Removal of pop members based on selecting members from normal distn centered on lowest fitness
#Reproduction of member in pop based on selecting a member from normal distn centered on highest fitness

import random
#import json
import numpy as np
import matplotlib.pyplot as plt
import math as m
import statistics as s
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
        s = 'ATG' #start codon
        for letter in range(3,r): #Set length of sequence
            num = random.randint(0,3)
            s += str(d[num])
        dna_seq_list.append(s)
        count += 1
    return dna_seq_list


def ran_prot_seq(r,n):
    """
    Function generates n random protein sequences of length r.
    """
    prot_d = {0:'M', 1:'A', 2:'V', 3:'P', 4:'N', 5:'R', 6:'H', 7:'W', 8:'Y', 9:'G', 10:'S', 11:'T', 12:'D', 13:'K', 14:'Q', 15:'I', 16:'L', 17:'E', 18:'F', 19:'C'}
    count = 0
    prot_s = ''
    prot_seq_list = []
    while count < n:
        prot_s = 'M'
        for letter in range(1,r):
            num = random.randint(0,19)
            prot_s += str(prot_d[num])
        prot_seq_list.append(prot_s)
        count += 1
    return prot_seq_list



#General fitness assignment function
#fit_dic = {'A':1, 'T':0,'G':1,'C':0} #Fix fitness values

def fit_dictionary(r):
    """
    Function makes fitness dictionary.
    Input - desired length of sequence (r)
    Output - Random assignment of 0 or 1 to each nucleotide at each position 0 - r
    Start codon positions - already assigned fixed fitnesses
    """
    nuc_list = ['A', 'G', 'T', 'C']
    fit_dic = {}  
    init_fit = {0:{'A':1,'G':0,'T':0,'C':0}, 1:{'A':0,'G':0,'T':1,'C':0},2:{'A':0,'G':1,'T':0,'C':0}}
    for j in range(0,3):
        for key,value in init_fit.items():
            if j == key:
                fit_dic[j] = value
    for i in range(3,r):
        nuc_dic = {}
        for letter in nuc_list:
            num = random.randint(0,1)
            nuc_dic[letter] = num
        fit_dic[i] = nuc_dic
    return fit_dic


def max_fitness(dic):
    """
    Function calculates maximum possible fitness for a sequence of length r
    Input - fitness dictionary calculated in fit_dictionary(r) function
    Calculated since, by chance, the fitness dictionary can assign 0 fitness to a position
    Hence, a maximum fitness can no longer be assumed to be 1
    """
    add = 0
    m = len(dic.keys())
    count = 0
    for key,value in dic.items():
        count = 0
        for num in value.values():
            if count < 1:
                if num == 1:
                    add += num
                    count += 1
    max_fit = round(add/m,3)
    return max_fit

def unique_sol(dic):
    """
    Function calculates total number of possible solutions which fulfill maximum fitness
    """
    mult = 1
    count = 0
    for key, value in dic.items():
        count = 0
        count_0 = sum(item == 0 for item in value.values())
        if count_0 < 4:
            for num in value.values():
                if num == 1:
                    count += 1
        elif count_0 == 4:
            count = 1
        mult *= count
    return mult

def fit_assign(func,seq_fit):
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
        fit_prob = round((1/len(seq)) * fit_sum,3) #calculates prob for seq
        #fit_per = round(fit_prob * 100,3)
        fit_assign_dic[seq] = fit_prob #assigns seq prob to dictionary
    return fit_assign_dic

prot_fit_dic = {'M':1, 'A':0, 'V':1, 'P':0, 'N':1, 'R':1, 'H':0, 'W':1, 'Y':1, 'G':0, 'S':1, 'T':0, 'D':1, 'K':1, 'Q':1, 'I':0, 'L':1, 'E':0, 'F':0, 'C':1}
def prot_fit_assign(prot_func,prot_fit_dic):
    prot_fit_assign_dic = {}
    for seq in prot_func: #loops over seq list generated by ran_dna_seq function
        fit_sum = 0
        fit_prob = 0
        for elem in seq: #loops over each seq element
            fit_sum += prot_fit_dic[elem] #sums each fitness assignment over seq
        fit_prob = round((1/len(seq)) * fit_sum,3) #calculates prob for seq
        prot_fit_assign_dic[seq] = fit_prob #assigns seq prob to dictionary
    return prot_fit_assign_dic

#Initial Fitness Assignment
#ran_seq = ran_dna_seq(21,10)
#ran_seq_fit = fit_assign(ran_seq,fit_dic) #set initial fitness of population

#n = 100 #number of initial sequences
#Kill member with lowest fitness
def remove_member(assign,pop_size):
    """
    Function removes sequences using probabilistic method.
    Sequence fitnesses normalised.
    Random number drawn from normal distribution centered on lowest fitness (variance 1)
    If member normal fitness greater than random num, it is not removed
    """
    assign_copy = assign.copy()
    num_mem = len(assign.values())
    #max_fit = max(assign.values())
    min_fit = min(assign.values())
    new_assign_dic = {}
    value_list = [] #list for calculating mean of sequence fitnesses
    new_assign = {} #final dictionary that is returned - pop with one member removed
    for value in assign.values():
        value_list.append(value)
    mean = s.mean(value_list) #find mean of input sequences
    var = round(s.variance(value_list),4) #find variance of input sequences
    norm_dic = {} #dictionary of normalised fitnesses
    if var > 0:
        min_fit_norm = round((min_fit - mean)/m.sqrt(var/num_mem),3) #normalise minimum fitness value
        for key,value in assign.items():
            norm_val = round((value - mean)/m.sqrt(var/num_mem),3) #normalise fitnesses
            norm_dic[key] = norm_val #create dictionary of normalised fitnesses
        #print(norm_dic)
        while int(len(new_assign_dic.values())) < int(pop_size - 1):
            #print(len(new_assign_dic.values()))
            st_norm = random.gauss(min_fit_norm,1)
            #print(st_norm)
            for key,value in norm_dic.items():
                if int(len(new_assign_dic.values())) < int(pop_size - 1):
                    if value > st_norm:
                        new_assign_dic[key] = value #if normalised fitness is greater than random number, it is appended to dictionary
        #print(new_assign_dic)
        for key,value in assign_copy.items():
            if key in new_assign_dic: #get back non-normalised fitness
                new_assign[key] = value
    else: #initialised if all seq have same fitness
        while int(len(new_assign.values())) < int(pop_size - 1):
            for key, value in assign_copy.items():
                if int(len(new_assign.values())) < int(pop_size - 1):
                    new_assign[key] = value 
    return new_assign
    

def bottleneck(population,reduction):
    pop_dic = {}
    new_pop_dic = {}
    pop_num = 0
    red_dic = {}
    for key in population.keys():
        pop_dic[key] = pop_num
        pop_num += 1
    size_pop = len(population.values())
    num_list = random.sample(range(0,size_pop),reduction)
    for key, value in pop_dic.items():
        if value not in num_list:
            red_dic[key] = value
    #print(red_dic)
    for key, value in population.items():
        if key in red_dic.keys():
            new_pop_dic[key] = value
    return new_pop_dic


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
def mutate_seq(mut_seq):
    """
    Function mutates members from input function. 
    One random mutation per sequence randomly along the sequence.
    """ 
    mutate_seq_list = [] 
    for key in mut_seq.keys():
        #print(key)
        list_key = list(key)
        #print(list_key)
        num = random.randint(3,len(list_key))
        #print(num)
        mut_num = random.randint(0,3)
        #print(mut_num)
        for i,j in enumerate(list_key):
            if i == num:
                j = dna_dic[mut_num]
                list_key[i] = j
               # print(list_key)
                string_key = "".join(list_key)
               # print(string_key)
                mutate_seq_list.append(string_key)
    return mutate_seq_list


prot_dic = {0:'M', 1:'A', 2:'V', 3:'P', 4:'N', 5:'R', 6:'H', 7:'W', 8:'Y', 9:'G', 10:'S', 11:'T', 12:'D', 13:'K', 14:'Q', 15:'I', 16:'L', 17:'E', 18:'F', 19:'C'}
def prot_mutate_seq(prot_seq):
    """
    Function randomly mutates a protein sequence
    """
    prot_mutate_seq_list = []
    for key in prot_seq.keys():
        prot_list_key = list(key)
        prot_num = random.randint(1,len(prot_list_key)) #start methionine never mutated
        prot_mut_num = random.randint(0,19)
        for i,j in enumerate(prot_list_key):
            if i == prot_num:
                j = prot_dic[prot_mut_num]
                prot_list_key[i] = j
                prot_string_key = "".join(prot_list_key)
                prot_mutate_seq_list.append(prot_string_key)
    return prot_mutate_seq_list


#Average A/G content
def aver_A_G_con(new_pop):
    """
    Function calculates average A/G content in pop.
    """
    seq_list = []
    A_count = 0
    G_count = 0
    seq_len = 0
    for key in new_pop.keys():
        seq_list.append(key)
    for seq in seq_list:
        slen = len(seq)
        seq_len += slen
        list_seq = list(seq)
        for elem in list_seq:
            if elem == 'A':
                A_count += 1
            elif elem == 'G':
                G_count +=1
    #print(A_count,G_count)
    aver_A_G = round(int(A_count + G_count)/int(seq_len),2)
    per_A_G_con = aver_A_G * 100
    #print(per_A_G_con)
    return round(per_A_G_con,2)



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
        i += 1 #next column
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

def total_uniq_sol(loop_num,r):
    """
    Function calculates total number of unique solitions for sequence of length r
    for different input fitness dictionaries
    """
    loop = 0
    sol_list = []
    loop_list = []

    while loop < loop_num:
        seq_fit = fit_dictionary(r)
        uniq_sol = unique_sol(seq_fit)
        print(uniq_sol)
        sol_list.append(uniq_sol)
        loop_list.append(loop)
        loop += 1
        x = loop_list
    y = sol_list
    #mean = s.mean(y)
    #print("mean is ",mean)
    plt.plot(x,y) ; plt.xlabel("Iteration") ; plt.ylabel("Number of Unique Solutions")
    plt.savefig("Fit_model_2_1_Num_Uniq_Sol_Sequence_Len_300_Fitness_Assign.png", dpi = 300)
        
    

def dna_fit_model(r,n,sim,pop_size,gen,reduction):
    """
    Function implements dna fitness model.
    Set length of sequence (r), pop size (n), number of simulations to run (sim), and 
    number of generations (gen)
    """
    num_sim = 0
    seq_fit = fit_dictionary(r)
    print("Fitness dictionary is ",seq_fit)
    max_fit = max_fitness(seq_fit)
    uniq_sol = unique_sol(seq_fit)
    #max_fit = []
    #min_fit = []
    #sim_round = {} 
    while num_sim < sim: #run simulation n times
#seq_len = []
        #print("simulation ", num_sim)
        gen_count = 0 #initial random sequences
        gen_dic_seq = {}
        gen_dic_fit = {}
        pop_seq = []
        pop_fit = []
        pop_size = 100
        h_dist_list = []
        init_seq = ran_dna_seq(r,n) #up to 100 seq per gen
        print("initial sequences are ",init_seq)
        init_fit = fit_assign(init_seq,seq_fit)
        for key, value in init_fit.items():
            pop_seq.append(key)
            pop_fit.append(value) 
        gen_dic_seq[gen_count] = pop_seq
        gen_dic_fit[gen_count] = pop_fit
        print("initial fitnesses are ",init_fit)  
        init_h_dist = hamming(init_fit,r)
        h_dist_list.append(init_h_dist)
        print("Initial pop hamming distance is ", init_h_dist)
        rem = remove_member(init_fit,pop_size)
        print("initial pop is ", rem)
        gen_count += 1
        #gen_ran = random.sample(range(0,gen),1)
        while gen_count < gen:   #initially run it for 10 generations
            pop_seq = []
            pop_fit = []
            rep = reproduction(rem)
            print("generation ", gen_count, " sequences to reproduce are ",rep)
            mut = mutate_seq(rep)
            print("generation ", gen_count, " new mutated sequences are ", mut)
            new_fit = fit_assign(mut,seq_fit)
            print("generation ", gen_count, " new sequence fitnesses are ", new_fit)
            rem.update(new_fit)
            print("rem update pop is ", rem)
            h_dist = hamming(rem,r)
            h_dist_list.append(h_dist)
            print("Generation ", gen_count, " hamming distance is ", h_dist)
            rem = remove_member(rem,pop_size)
            print("generation ", gen_count, " new population is ", rem)
            if gen_count == 100:
                bot = bottleneck(rem,reduction)
                rem = bot
                pop_size = 50
            for key, value in rem.items():
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
        for key in gen_dic_fit.keys():
            x.append(key)
        y = []
        i = 0
        for key,value in gen_dic_fit.items():
            if key == i:
                mean = round(s.mean(value),3)
            y.append(mean)
            i += 1
    
   # sim_round[num_sim] = gen_dic_fit
        #plt.plot(x,y) ; plt.xlabel("Generation"); plt.ylabel("Average Fitness"); plt.ylim(top=max_fit)
        print(max_fit)
        print(uniq_sol)
        #print(output)
        #Graph for average G/C content
        #plt.plot(x,A_G_con_list) ; plt.xlabel("Generation"); plt.ylabel("Average A/G Content (%)")
        plt.plot(x,h_dist_list) ; plt.xlabel("Generation") ; plt.ylabel("Average Hamming Distance")
        num_sim +=1

    #plt.savefig("Fit_model_2_1_bottleneck_at_gen_100_50_seq_left_5_sims_len_150_100_seq_per_gen_1000_gen.png", dpi=300)
    #plt.savefig("Fit_model_2_5_sims_A_G_con_len_300_10_seq_per_gen_100_gen_T_and_C_0_fit.png", dpi=300)
    plt.savefig("Fit_model_2_1_bottleneck_at_gen_100_5_sims_Hamming_distance_len_300_100_seq_per_gen_1000_gen.png", dpi=300)
    #plt.savefig("Fit_model_2_1_sim_Fitness_Hamming_Dist_len_99_100_seq_init_10_seq_per_gen_1000_gen_T_and_C_0_fit.png", dpi=300)
