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

class Trie_Autocomplete():
    head={}
    def add_word(self,word):
        curr= self.head
        for ch in word:
            if ch not in curr:
                curr[ch]={}
            curr=curr[ch]
        curr['*']=True
    def search_word(self,word):
        possible_words=[]
        pos_word=""
        base_word=""
        curr=self.head
        for ch in word:
            if ch not in curr:
                return ('No such word exist')
            base_word+=ch
            curr=curr[ch]
        pos = curr
        for i in curr.keys():
            while i != '*':
                pos_word += i
                pos = pos[i]
                i= list(pos.keys())[0]
            possible_words.append(base_word+pos_word)
            pos_word=""
            pos=curr
        print(possible_words)
        if '*' in curr:
            return True
        else: return False

a=Trie_Autocomplete()
b=['rohan','rohani','sneha','rita','pramod','ghunghuru','sanjay','vidhya','rajat','mohak','pavan','gurpreet','sanampreet','mohit']
for i in b:
    print(i)
    a.add_word(i)
print(a.search_word('rohan'))

#0,1,1,2,3,5,8
#Q1how you find a number at aparticular position of afibonaci series
#Q given a fibonacci number write aprograme to find the position

def feb(pos):
    if pos < 2:
       return pos
    else:
        return feb(pos-2)+feb(pos-1)
def print_fibonacci(pos):
    arr=[feb(i) for i in range(pos+1)]
    print(arr)
print_fibonacci(7)



a=[4,1,5,2,3,0,10]

def max_heapify(list):
    for i in range(1,len(list)):
        left= 2*i
        right=(2*i+1)
        largest=i
        if left <= len(list) and list[left-1] > list[i-1]:
            largest=left
        if right <= len(list) and list[right-1] > list[i-1]:
            if largest==None or (list[right-1]> list[left-1]):
                largest=right
            else:
                largest=left
        if largest != i:
            list[i-1],list[largest-1]=list[largest-1],list[i-1]
        return list

def find_k_smallest_element(a,element):
    heap=a[:element]
    heap=max_heapify(heap)
    for i in range(element,len(a)):
        if a[i] < heap[0]:
            heap[0]=a[i]
            heap=max_heapify(heap)
    print(heap)

find_k_smallest_element(a,3)

# Check valid parenthisis...
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        dict_ = {'(': ')',
                 '{': '}',
                 '[': ']'}
        if len(s) % 2 == 1:
            return False
        if len(s) ==0:
            return True
        else:
            stack=[s[:1] for i in s[0] if i in dict_]
            if len(stack)==0:
                return False
            i=1
            while i < len(s):
                if len(stack) == 0:
                    stack.append(s[i])
                    i+=1
                if stack[-1] in dict_:
                    if s[i] != dict_[stack[-1]]:
                        stack.append(s[i])
                    elif s[i]==dict_[stack[-1]]:
                        stack.pop(-1)
                    i+=1
                else:
                    return False
            if len(stack)==0:
                return True
            else:
                return False

#Palindrom substring Manchers algorithim ---------------needs improvement
class Solution(object):
    def _longestPalindrome(self,width,boundary,charecter,pos,leftpos,rightpos):
        if leftpos < 0:
            boundary = (leftpos,rightpos)
            leftpos = leftpos + 1
            self._longestPalindrome(width, boundary, charecter, pos, leftpos, rightpos)
        elif rightpos == len(charecter):
            boundary = (leftpos, rightpos)
            rightpos=rightpos-1
            self._longestPalindrome(width, boundary, charecter, pos, leftpos, rightpos)
        elif charecter[leftpos]==charecter[rightpos]:
            boundary=(leftpos,rightpos)
            if leftpos==0 and rightpos == len(charecter)-1:
                width[charecter[pos]] = [leftpos , rightpos , abs(rightpos - leftpos) + 1]
                return (width,boundary)
            width[charecter[pos]] =[leftpos,rightpos,abs(rightpos-leftpos)+1]
            self._longestPalindrome(width,boundary,charecter,pos,leftpos-1,rightpos+1)
        elif charecter[leftpos]!=charecter[rightpos]:
            if boundary[0] < 0:
               if abs((boundary[0]+1) - (rightpos-1))==0:
                   width[charecter[pos]] = [boundary[0]+1, rightpos-1, 1]
               else:
                   width[charecter[pos]] = [boundary[0] + 1, rightpos - 1, abs((boundary[0]+1) - (rightpos-1)) + 1]
               boundary = (boundary[0] + 1, rightpos - 1)
               return (width, boundary)
            elif boundary[1] == len(charecter):
               if abs((boundary[1]-1) - (leftpos+1))==0:
                   width[charecter[pos]] = [leftpos + 1, boundary[1]-1, 1]
               else:
                   width[charecter[pos]] = [leftpos + 1, boundary[1]-1, abs((boundary[1]-1) - (leftpos + 1)) + 1]
               boundary = (leftpos+1, boundary[1] - 1)
               return (width, boundary)
            else:
                boundary=(leftpos+1,rightpos-1)
                if (rightpos-1) - (leftpos+1)==0:
                    width[charecter[pos]]=[leftpos+1,rightpos-1,1]
                else:
                    width[charecter[pos]] = [leftpos+1, rightpos-1, abs((rightpos-1) - (leftpos+1))+1]
                return (width,boundary)

    def longestPalindrome(self, s):
        stack=[]
        outbound=0
        center=""
        boundary=(0,0)
        max_bound=[]
        i=0
        width={}
        while i <= len(s)-1:
            if s[i] not in width:
                center = s[i]
                self._longestPalindrome(width,boundary,s,i,i-1,i+1)
                print(width)
                if len(max_bound)==0:
                    max_bound.append(width[s[i]][0])
                    max_bound.append(width[s[i]][1])
                else:
                    if (width[s[i]][1] - width[s[i]][0]) > (max_bound[1]-max_bound[0]):
                        max_bound[0]=width[s[i]][0]
                        max_bound[1]=width[s[i]][1]
                stack.append(width[s[i]][2])
            else:
                for j in range(i,max_bound[1]+1):
                    if  width[s[j]][0] >= max_bound[0]  and width[s[j]][1]<= max_bound[1]:
                        stack.append(width[s[j]][2])
                    elif width[s[j]][0] < max_bound[0] and width[s[j]][1]<= max_bound[1]:
                        outbound = width[s[j]][2] - ((max_bound[0] - width[s[j]][0]) + 1)
                        stack.append(outbound)
                if max(stack[i:max_bound[1]+1]) != outbound:
                    repos =stack[i:max_bound[1]+1].index(max(stack[i:max_bound[1]+1]))+i
                    center = s[repos]
                    self._longestPalindrome(width, boundary, s, repos, repos - 1, repos + 1)
                    if (width[center][1] - width[center][0]) > (max_bound[1] - max_bound[0]):
                        max_bound[0] = width[center][0]
                        max_bound[1] = width[center][1]
                    stack[repos] = width[center][2]
                    i = len(stack) - 1
                else:
                    repos=len(stack)
                    center = s[repos]
                    self._longestPalindrome(width, boundary, s, repos, repos - 1, repos + 1)
                    i=len(stack)
            i+=1
        print(max_bound,stack)
b='abaxabaxabybaxabyb'
#b='xabaxy'
check= Solution()
check.longestPalindrome(b)

-- BST for sorted array
class Node():
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None
class Solution(object):
    def __init__(self):
        self.root = None
    def create_bs(self,currpos,midvalue,start,end,array):
        if start > end:
            currpos.value=None
            return
        left_tree_start= start
        left_tree_end = midvalue-1
        right_tree_end = end
        right_tree_start = midvalue + 1
        # left child creation-----
        left_tree_mid = int((left_tree_start + left_tree_end) / 2)
        currpos.left_child = Node(array[left_tree_mid])
        self.create_bs(currpos.left_child, left_tree_mid, start, left_tree_end, array)
        # right child creation-----
        right_tree_mid = int((right_tree_start + right_tree_end) / 2)
        currpos.right_child = Node(array[right_tree_mid])
        self.create_bs(currpos.right_child,right_tree_mid,right_tree_start,right_tree_end,array)

    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        start = 0
        end = len(numbers) - 1
        mid = int((start + end) / 2)
        if self.root == None and len(numbers) > 0:
            self.root = Node(numbers[mid])
            self.create_bs(self.root, mid, start, end, numbers)
a=Solution()
a.twoSum([0,1,2,3,6,10,12,15,17],5)

class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        value = numbers[:]
        start = 0
        end = len(numbers) - 1
        while start != end:
            if numbers[end] + numbers[start] > target:
                value.pop(end)
                end = end - 1
            elif numbers[end] + numbers[start] < target:
                value.pop(0)
                start=start+1
            else:
                return [start, end]
        return value
a=Solution()
a.twoSum([1,2,3,4,4,9,56,90],8)


# Search in rotated sorted arrays.
class Node():
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None


class Solution(object):
    def __init__(self):
        self.root = None

    def create_bs(self, currpos, midvalue, start, end, array):
        if start > end:
            currpos.value = None
            return
        left_tree_start = start
        left_tree_end = midvalue - 1
        right_tree_end = end
        right_tree_start = midvalue + 1
        # left child creation-----
        left_tree_mid = int((left_tree_start + left_tree_end) / 2)
        currpos.left_child = Node(array[left_tree_mid])
        self.create_bs(currpos.left_child, left_tree_mid, start, left_tree_end, array)
        # right child creation-----
        right_tree_mid = int((right_tree_start + right_tree_end) / 2)
        currpos.right_child = Node(array[right_tree_mid])
        self.create_bs(currpos.right_child, right_tree_mid, right_tree_start, right_tree_end, array)

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        sort = []
        index = 0
        for i, v in enumerate(nums):
            if v > nums[i + 1]:
                index = i
                break
        sort = nums[index + 1:] + nums[:index + 1]
        start = 0
        end = len(sort) - 1
        mid = int((start + end) / 2)
        if self.root == None:
            self.root = Node(sort[mid])
            self.create_bs(self.root, mid, start, end, sort)
a= Solution()
b=[4,5,6,7,0,1,2]
a.search(b,0)

#number of subarrays whoes consequtive sum is k for positive numbers
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        i = 0
        stack = []
        output = []
        while i < len(nums):
            if i == 0:
                stack.append(nums[i])
                if sum(stack) == k:
                    output.append(stack)
                i += 1
            else:
                stack.append(nums[i])
                i += 1
                if sum(stack) == k:
                    output.append(stack)
                    stack = [stack[-1]]
                elif sum(stack) > k:
                    stack = [stack[-1]]
                    if stack[0] == k:
                        output.append(stack[0])
        print(len(output))
        return (len(output))

a=Solution()
b=[-1,-1,1]
a.subarraySum(b,2)
#-- Obstacle matrix path ---
def obstacle_path(a):
    if len(a)==0:
     return 0
    paths = [[0]*len(a[0]) for i in a]
    print(paths)
    if a[0][0] == 1:
        paths[0][0] = 1
    for i in range(1, len(a)):
        if a[i][0] == 1:
            paths[i][0] = paths[i-1][0]
    for j in range(1, len(a[0])):
        if a[0][j] == 1:
            paths[0][j] = paths[0][j-1]
    for i in range(1, len(a)):
        for j in range(1, len(a[0])):
            if a[i][j] == 1:
                paths[i][j] = paths[i-1][j] + paths[i][j-1]
    return paths[-1][-1]

print(obstacle_path(a))
#---------------------------------------------------
#---- Sum of subarrays equals to k--------------
class Solution(object):
    def subarraySum(self, nums, k):
        count = 0
        sum_index = 0
        dict_look = {}
        sum_left = 0
        for i, v in enumerate(nums):
            sum_index = sum_index + v
            sum_left = sum_index - k
            if sum_left in dict_look:
             count += dict_look[sum_left]
            if sum_index in dict_look:
             dict_look[sum_index] = dict_look[sum_index]+1
            else:
             dict_look[sum_index]=1
            if sum_index==k:
             count+=1

        print(count)
b=[0,0,0]
c=0
a=Solution()
a.subarraySum(b,c)
#----------------------------------Data challange @ gust--------------------
import numpy as np
import pandas as pd
a=[]
f = open("C:/Users/rohan/Desktop/uic/data challange/data1.txt", "r")# copied the data created a text file of it and created a data frame
for x in f:
   a.append(x.split("\t"))
npa= np.array(a)
df = pd.DataFrame(npa,columns=['id', 'created_at', 'user_id', 'amount'])
df['amount']=df['amount'].apply(lambda x: x[:-1])
df['created_at']=df['created_at'].apply(lambda x: x[:-7])
df.set_index("id")
df.to_csv("C:/Users/rohan/Desktop/uic/data challange/data_to_export")
print(df)



npa= np.array(a)
df = pd.DataFrame(npa,columns=['id', 'created_at', 'user_id', 'amount'])
df['amount']=df['amount'].apply(lambda x: x[:-1])
df['amount']=pd.to_numeric(df['amount'])
df['created_at']=df['created_at'].apply(lambda x: x[:10])
df['created_at']=pd.to_datetime(df['created_at'])
df['month']=df['created_at'].apply(lambda x: x.month)
df['day']=df['created_at'].apply(lambda x: x.day)
df.set_index("id")
df.to_csv("C:/Users/rohan/Desktop/uic/data challange/data_to_export")
def number_of_days(df,ndays=0):
    dicti={}
    for i in np.unique(df['user_id']):
        min_date=min(df[df['user_id']==i]['created_at'])
        max_date=max((df[df['user_id']==i]['created_at']))
        diff = (max_date-min_date)
        if ndays==0 and diff==0:
            dicti[i]=1
        if int(diff.days) >= ndays:
            dicti[i]=int(diff.days)
    return  dicti
def revenue_cust(custbydays,df,ndays=0):
    dictav={}
    for i in list(custbydays.keys()):
        min_date=min(df[df['user_id']==i]['created_at'])
        new_date= min_date + datetime.timedelta(days=ndays)
        sumrev=df[(df['created_at'] <=new_date) & (df['user_id']==i)]['amount'].sum()
        dictav[i]=round(sumrev/custbydays[i],2)
    return dictav
custbydays=number_of_days(df,ndays)
averagerev_dict=revenue_cust(custbydays,df,ndays)
final_df=pd.DataFrame()
final_df['customer']=custbydays.keys()
final_df['days']=custbydays.values()
final_df['Average revenue per customer']=averagerev_dict.values()
final_df.set_index("customer")
print(final_df)
#------------------------- Distinct subset in an array---------------------------------------------
class Solution(object):
    subset_dict={}
    def subsetsWithDup_(self, arr, pos, subset_dict, endpos,set,subset_ar):
        if pos == endpos:
            a = []
            for i in set:
                if i != None:
                    a.append(i)
            if len(a) > 0 and tuple(a) not in subset_dict:
                subset_dict[tuple(a)]='one'
                subset_ar.append(a)
            return
        set[pos]=arr[pos]
        self.subsetsWithDup_(arr,pos+1,subset_dict,endpos,set,subset_ar)
        set[pos]=None
        self.subsetsWithDup_(arr,pos+1,subset_dict,endpos,set,subset_ar)

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        set= [None]*len(nums)
        subset_ar = []
        subset_ar.append([])
        start_pos = 0
        end_pos = len(nums)
        self.subsetsWithDup_(nums, start_pos, self.subset_dict, end_pos,set,subset_ar)
        print(subset_ar)
        return(subset_ar)
a=Solution()
b=[1,1]
a.subsetsWithDup(b)

class Solution(object):
    def nextGreaterElement(self, findNums, nums):
        """
        :type findNums: List[int]
        :type nums: List[int]
        :rtype: List[int]
        """
        output=[]
        for i in findNums:
            ar = nums[nums.index(i):]
            if len(ar)==1 or i==max(ar):
                output.append(-1)
            else:
                for v in ar[1:]:
                    if i < v:
                        output.append(v)
                        break
        print(output)
        return output

a=Solution()
b=[[4,1,2],
[1,2,3,4]
]
a.nextGreaterElement(b[0],b[1])
