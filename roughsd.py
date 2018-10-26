import numpy as np
# First Question-------------------
def print_int(array,value):
    final_len = 0
    a= np.unique(array)
    for i,v in enumerate(a):
        for value in range(i,len(a)):
            if (a[i]+ a[value]==k):
                final_len +=1
    return final_len

#a= [6,1,3,46,1,3,9,47]
#k=10
#print(print_int(a,k))
# second Question Returning chocolate

def return_chocholate(value):
    quotient,remainder = divmod(value,3)
    array_len = quotient + remainder
    final_answer = []
    for i in range(quotient):
        if (remainder == 0 and i==0):
            final_answer.append(1)
        else:
            final_answer.append(array_len)
        array_len =array_len + 2
    final_answer.append(1)
    return (sum(final_answer)%(10**9+value))

#print(return_chocholate(20))

def minimumBribes(q):
    bribes = 0
    for i in range(len(q)-1,-1,-1):
        if q[i] - (i + 1) > 2:
            print('Too chaotic')
            return
        for j in range(max(0, q[i] - 2),i):
            if q[j] > q[i]:
                bribes+=1
    print(bribes)
#q=[[1, 2, 5, 3, 7, 8, 6, 4,],[5, 1, 2, 3, 7, 8, 6, 4],[1, 2, 5, 3, 4, 7, 8, 6,],[2, 1, 5, 3, 4,],[2, 5, 1, 3, 4,]]
#for i in q:
 #   minimumBribes(i)

def bubble_sort(arr):
    #implementing bubble sort..
    pos_shift=len(arr)-1
    swap=0
    swap_temp=0
    is_sorted = False
    while is_sorted == False:
        is_sorted = True
        for i in range(pos_shift):
            if(arr[i] > arr[i+1]):
                if swap_temp!=arr[i]:
                    swap_temp=arr[i]
                    swap+=1
                arr[i],arr[i+1]=arr[i+1],arr[i]
                is_sorted = False
        pos_shift=pos_shift-1
    print(arr)
    print(swap)

b=[2,3,1,4,5]
#print(a.index(1))
#bubble_sort(a)

def minimumSwaps(arr):
    c_arr= []
    swap=0
    for i in range(len(arr)):
        minvalue= min(arr[i:])
        indexof =arr.index(minvalue)
        if indexof != i:
            arr[i],arr[indexof]=arr[indexof],arr[i]
            swap=swap+1
    print(swap)
#minimumSwaps(a)
b=[2,3,4,1,5]
def minimumSwaps1(arr):
    c_arr= sorted(arr)
    swap=0
    ind_dict={v:i for i,v in enumerate(arr)}
    for i,v in enumerate(arr):
        corrected_value = c_arr[i]
        if v!= c_arr[i]:
            index_of= ind_dict[corrected_value]
            #print(index_of)
            arr[index_of],arr[i]= arr[i],arr[index_of]
            ind_dict[v]=index_of
            ind_dict[corrected_value]=i
            swap+=1
    return(swap)
#minimumSwaps1(a)

def postion_same_assign(value0,value1,array,to_value):
    for i in range(value0-1,value1):
        array[i]=to_value
    return array


b = [[2, 6, 8], [3, 5, 7], [1, 8, 1], [5, 9, 15]]
c = [0] * 10
temp=0
max_value=0
# print((c[b[0]-1:b[1]]==[b[2]]))
for i, v in enumerate(b):
    #print(i, v)
    if i == 0:
        postion_same_assign(v[0],v[1],c,v[2])
    else:
        for inside in range(v[0],v[1]+1):
            c[inside - 1] = c[inside - 1] + v[2]
            temp=c[inside - 1]
            if(temp > max_value):
                max_value=temp
#print(max_value)

s='pwwkew'
s1='abcabcbb'
s2=['bbbb']
s3="aab"


def lengthOfLongestSubstring(s):
    dicts = {}
    maxlength = start = 0
    for i, value in enumerate(s):
        if value in dicts:
            sums = dicts[value] + 1
            if sums > start:
                start = sums
        num = i - start + 1
        if num > maxlength:
            maxlength = num
        dicts[value] = i
    return maxlength

s='1432219'
s1='10200'
s2='112'
s3='5337'
s4="10"
def removeKdigits(num, k):
        start = 0
        output = ""
        num_dict = {i: v for i, v in enumerate(num)}
        print(num_dict)
        zeros=0
        for i, value in enumerate(num):
            if len(num) == k:
                return ("0")
            if start >= k:
                break
            else:
                for j in range(k):
                    key = i + j + 1
                    if (k-start)+i >= len(num):
                        num_dict[i] = 'r'
                        #print("exception")
                    else:
                        if key < len(num) and value > num_dict[key]:
                            num_dict[i] = 'r'
                            start+=1
                            break
        for i,(x,v) in enumerate(sorted(num_dict.items(),key= lambda x:x[0])):
            if v != "0" and v != 'r':
                output = output + str(v)
                zeros+=1
            else:
                if zeros > 0 and v != 'r':
                    output = output + str(v)
        if output=="":
            output = [v for i,v in enumerate(num_dict.values()) if v !='r']
            output=''.join(output)[:1]
        print(num_dict)
        return(output)


#print(removeKdigits(s,555))
#print(removeKdigits(s1,1))
#print(removeKdigits(s2,1))
#print(removeKdigits(s3,2))
#print(removeKdigits(s4,1))

def removeKdigits1(num, k):
    stack = []
    for x in num:
        while stack and x < stack[-1] and k > 0:
            stack.pop()
            k -= 1
        stack.append(x)
    for i in range(k):stack.pop()
    while stack and stack[0] == '0': del stack[0]
    if not stack:stack.append('0')
    return ''.join(stack)
#print(removeKdigits1(s,3))


s= "the     sky     is     blue"
def reverseWords(s):
    output=""
    value= [v for i,v  in enumerate(s.split(" ")) if len(v) > 0 ]
    value.reverse()
    for i,v in enumerate(value):
        output=output + v +" "
    print(output)
#reverseWords(s)

a=[(1, 10),(2, 7),(10, 20),(11,30),(8,12),(3,19)]
def minMeetingRooms(intervals):
        # step 1 sorting array with the start time
        #sort_intervals = [(v.start, v.end) for i, v in enumerate(intervals)]
        sort_intervals = sorted(intervals, key=lambda x: x[0])
        # step 2 procced with rest of part
        dict_room = {}
        room_count = 0
        minimum = {}
        min_key = 0
        for i, (start, end) in enumerate(sort_intervals):
            if i == 0:
                room_count += 1
                dict_room[room_count] = [start, end]
                minimum['Value'] = [start, end]
                min_key = room_count
            else:
                if start < minimum['Value'][1]:
                    room_count += 1
                    if end < minimum['Value'][1]:
                        minimum['Value'] = [start, end]
                        min_key = room_count
                        dict_room[min_key] = [start, end]
                    else:
                        dict_room[room_count] = [start, end]
                elif start > minimum['Value'][1]:
                    dict_room[min_key] = [start, end]
                    for c,(key_, val) in enumerate(sorted(dict_room.items(), key=lambda x: x[1][1])):
                        if c == 0:
                            min_key = key_
                            minimum['Value'] = val
                            break
                elif start == minimum['Value'][1]:
                    dict_room[min_key] = [start, end]
                    for c,(key_, val) in enumerate(sorted(dict_room.items(), key=lambda x: x[1][1])):
                        if c == 0:
                            min_key = key_
                            minimum['Value'] = val
                            break
        print(room_count)
#minMeetingRooms(a)

version1 = "1.0.1"
version2 = "1"
def compareVersion(version1, version2):
        value1 = version1.split('.')
        value2 = version2.split('.')
        same=0
        value1 =[int(i)for i in value1]
        value2 =[int(i) for i in value2]
        maxlen=max(len(value1),len(value2))
        for i,v in zip(value1,value2):
            if int(i) > int(v):
                return(1)
            elif int(v) > int(i):
                return(-1)
            else:
                same=0
        if len(value1)!=len(value2):
            if maxlen==len(value1) and max(value1[len(value2):])!=0:
                return(1)
            elif maxlen==len(value2) and max(value2[len(value1):])!=0:
                return (-1)
            else:
                return 0
        return 0
#print(compareVersion(version1,version2))

#print(int('000000000123'))

import random
import re
same='AAACCCAATTTACACAGCTGGGCCCAGTGGGGGGGGG'
a= ['AGTGGGGGGGGG','AAACCCAATTT','TTTACACAGCT','GCTGGGCCCAGT']
# Complete the analyze_dna function below.
def analyze_dna(strands, codon_mapping= None):
    valid_strands = [i for i in strands if len(i) > 10 and len(i) < 100 and i.isupper()==True]
    if len(valid_strands) < 3:
        return None
    dict_t={v[:3]:i for i,v in enumerate(valid_strands)}
    temp=None
    count = 0
    for i,v in enumerate(valid_strands):
        value= v[-3:]
        if value in dict_t.keys():
            if count==0:
                temp=(v[:-3] + valid_strands[dict_t[value]])
                count+=1
            else:
                temp=temp[:-3]+ valid_strands[dict_t[value]]
    print(temp)
    n=3
    split=[temp[i:i+n] for i in range(0, len(temp), n)]
    print(split)
#analyze_dna(a)


class Solution(object):
    def dfs(self, row, column, grid,count):
       if len(count)==0:
           return
       if column == len(grid[0]) - 1 and row == len(grid) - 1:
            grid[row][column] = 0
            count.pop(count.index((row, column)))
            self.dfs(count[-1][0], count[-1][1], grid, count)
       elif column == len(grid[0]) - 1 and int(grid[row + 1][column]) == 0:
            grid[row][column] = 0
            count.pop(count.index((row, column)))
            self.dfs(count[-1][0],count[-1][1],grid,count)
       elif row == len(grid) - 1 and int(grid[row][column + 1]) == 0:
            grid[row][column] = 0
            count.pop(count.index((row, column)))
            self.dfs(count[-1][0], count[-1][1], grid, count)
       elif int(grid[row + 1][column]) == 0 and int(grid[row][column + 1]) == 0:
           grid[row][column] = 0
           count.pop(count.index((row,column)))
           if len(count)==0:
               return
           self.dfs(count[-1][0], count[-1][1], grid, count)

       else:
           if column == len(grid[0]) - 1 and int(grid[row + 1][column]) == 1:
               count.append((row + 1, column))
               self.dfs(row + 1, column, grid, count)
           elif row == len(grid) - 1 and int(grid[row][column + 1]) == 1:
                count.append((row, column+1))
                self.dfs(row, column + 1, grid,count)
           elif int(grid[row + 1][column]) == 1 and int(grid[row][column + 1]) == 1:
                count.append((row, column + 1))
                self.dfs(row, column + 1, grid,count)
           elif int(grid[row + 1][column]) == 0 and int(grid[row][column + 1]) == 1:
                count.append((row, column + 1))
                self.dfs(row, column + 1, grid,count)
           elif int(grid[row + 1][column]) == 1 and int(grid[row][column + 1]) == 0:
                count.append((row+1, column))
                self.dfs(row + 1, column, grid,count)



    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        island = 0
        for row in range(len(grid)):
            for column in range(len(grid[0])):
                count = []
                if int(grid[row][column]) == 1:
                    island += 1
                    count.append((row,column))
                    self.dfs(row, column, grid,count)
        print(island)
a=Solution()
a.numIslands(grid=[["1","1","1","1","1","1"],["1","1","0","0","0","1"],["1","0","1","0","0","1"],["1","0","1","0","0","1"],["1","1","0","1","0","1"]])


